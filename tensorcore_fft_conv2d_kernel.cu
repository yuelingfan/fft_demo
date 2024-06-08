#include <stdio.h>
#include "tensorcore_fft_conv2d.h"
#include <math.h>

#define assert(X)             \
    if (!(X))                 \
    {                         \
        printf("Stopped!\n"); \
        return;               \
    }
#define LL long long int

using namespace nvcuda;
const int WARP_SIZE = 32, WMMA_M = 16, WMMA_N = 16, WMMA_K = 16, CONT_SIZE = 32;

__device__ inline void complex_mul(wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> &frag_F_real, wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> &frag_F_imag,
                                   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> &frag_in_real, wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> &frag_in_imag,
                                   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> &frag_out_real, wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> &frag_out_imag)
{
    wmma::fill_fragment(frag_out_real, 0.0);
    wmma::fill_fragment(frag_out_imag, 0.0);

    wmma::mma_sync(frag_out_real, frag_F_imag, frag_in_imag, frag_out_real);
    for (int i = 0; i < frag_out_real.num_elements; i++) frag_out_real.x[i] = -frag_out_real.x[i];
    wmma::mma_sync(frag_out_real, frag_F_real, frag_in_real, frag_out_real);

    wmma::mma_sync(frag_out_imag, frag_F_real, frag_in_imag, frag_out_imag);
    wmma::mma_sync(frag_out_imag, frag_F_imag, frag_in_real, frag_out_imag);
}

__device__ inline void complex_mul_acc(wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> &frag_F_real, wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> &frag_F_imag,
                                       wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> &frag_in_real, wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> &frag_in_imag,
                                       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> &frag_out_real, wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> &frag_out_imag)
{
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_buf_real;
    wmma::fill_fragment(frag_buf_real, 0.0);

    wmma::mma_sync(frag_buf_real, frag_F_imag, frag_in_imag, frag_buf_real);
    for (int i = 0; i < frag_buf_real.num_elements; i++) frag_buf_real.x[i] = -frag_buf_real.x[i];
    wmma::mma_sync(frag_buf_real, frag_F_real, frag_in_real, frag_buf_real);
    for (int i = 0; i < frag_buf_real.num_elements; i++) frag_out_real.x[i] += frag_buf_real.x[i];

    wmma::mma_sync(frag_out_imag, frag_F_real, frag_in_imag, frag_out_imag);
    wmma::mma_sync(frag_out_imag, frag_F_imag, frag_in_real, frag_out_imag);
}

__device__ __host__ inline half2 W_N_K(int N, int K, int fft)
{
    // k索引，N点fft
    // https://zhuanlan.zhihu.com/p/473783874

    // fft = 1 -> fft, fft = -1 -> ifft
    half2 t;
    if (fft == 1)
    {
        t.x = cosf(2 * M_PI * K / N);
        t.y = -sinf(2 * M_PI * K / N);
    }
    else
    {
        t.x = cosf(2 * M_PI * K / N);
        t.y = sinf(2 * M_PI * K / N);
    }
    return t;
}

__device__ __host__ inline float2 W_N_K_fp32(int N, int K, int fft)
{
    // fft = 1 -> fft, fft = -1 -> ifft
    float2 t;
    if (fft == 1)
    {
        t.x = cosf(2 * M_PI * K / N);
        t.y = -sinf(2 * M_PI * K / N);
    }
    else
    {
        t.x = cosf(2 * M_PI * K / N);
        t.y = sinf(2 * M_PI * K / N);
    }
    return t;
}

__device__ inline half2 const cmul(const half2 &a, const half2 &b) // 复数乘法
{
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

__device__ inline half2 const cmul_mixed(const half2 &a, const float2 &b) { return {a.x * __float2half(b.x) - a.y * __float2half(b.y), a.x * __float2half(b.y) + a.y * __float2half(b.x)}; }
__device__ inline float2 const cmul_full(const float2 &a, const float2 &b) { return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }

__device__ inline void swap(half &a, half &b)
{
    half tmp = a;
    a = b;
    b = tmp;
}
__device__ inline void swap(double &a, double &b)
{
    double tmp = a;
    a = b;
    b = tmp;
}
__device__ inline int swapnum_line(int y, int line) { return y - line - 1; }
__device__ inline int swapnum_ele(int x, int ele) { return x - ele - 1; }

//**对基256而言，第一阶段基16，第二阶段基16，仅有两个阶段。**
// 相比之下，512有三个阶段，分别是16，16，2.
template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_256_0(half2 *in, half *F_real, half *F_imag, int fft)
{
    extern __shared__ half2 smem_in[];

    int t_block = threadIdx.x + threadIdx.y * blockDim.x;  // thread id in its block
    LL block_start = (LL)blockIdx.x * 256 * (LL)CONT_SIZE; // cont_size = 32 ??

    // www.eet-china.com/mp/a42816.html
    // https://zhuanlan.zhihu.com/p/353208013
    // m*k * k*n = m*n
    // 左乘DFT矩阵，所以此处DFT矩阵为行优先row_major
    //**此处的row_major和col_major实际上代表了本身的我们所设计的input的layout。
    //**程序由此确定tile中的元素是按行从上往下存取，还是按列从左往右存取（load_matrix_sync,
    // store_matrix_sync）。
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major> frag_F_real; // 16*16*16
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16); // load to fragment
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_col = threadIdx.x / 16 * 4 + threadIdx.x % 16 / 8 * 8 + threadIdx.x % 4;
    half2 twiddle_unit = W_N_K(256, raw_col, fft); // 旋轉因子

    // 一个warp负责16*16，则一个含有8个（NUM_WARP）的block负责8*16*16个元素处理。
    // 总体而言，对单个layer，最多处理256*CONT_SIZE个元素(疑问，如果数据太长，还能处理完吗？)
    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 16 * 16)
    {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_tmp0;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_tmp1;

        // 注意这里，threadIdx.y代表的是warpID，因为thread = {32，NUM_WARP}
        int warp_start = i + threadIdx.y * 256; // 16*16 matrix = 256 element strid
        // 每个取16个，一共32个
        wmma::load_matrix_sync(frag_in_tmp0, (half *)(in + block_start + warp_start),
                               32); // 因为是col_major，所以32是列元素数，实虚相间
        wmma::load_matrix_sync(frag_in_tmp1, (half *)(in + block_start + warp_start) + 16,
                               32); // 加16是为了避免bank conflict。16 = 8个复数

        for (int j = 0; j < 8; ++j)
        {
            frag_in_real.x[j] = frag_in_tmp0.x[2 * j];
            frag_in_imag.x[j] = frag_in_tmp0.x[2 * j + 1];
            frag_in_real.x[8 + j] = frag_in_tmp1.x[2 * j];
            frag_in_imag.x[8 + j] = frag_in_tmp1.x[2 * j + 1];
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        //+256同样也是避免bank conflict的手段。因为下位采用+256 load_matrix_sync
        wmma::store_matrix_sync((half *)(smem_in + warp_start), frag_out_real, 16, wmma::mem_row_major);
        wmma::store_matrix_sync((half *)(smem_in + warp_start) + 256, frag_out_imag, 16,
                                wmma::mem_row_major); // 合并后，这256个复数作为一个整体。所以，前256实部，后256虚部

        wmma::load_matrix_sync(frag_in_real, (half *)(smem_in + warp_start), 16);
        wmma::load_matrix_sync(frag_in_imag, (half *)(smem_in + warp_start) + 256, 16);

        //**时刻注意
        //**这是warp-level的操作！
        //**在此处，对16*16的元素乘旋转因子（点积）
        half2 twiddle_factor = {1.0, 0};
        for (int j = 0; j < 16; ++j)
        {
            int row = j;
            int col = raw_col; // 此warp中的每个thread所存取的fragment对应于一个tile中的列都各不相同
            half2 in_ele = {frag_in_real.x[j], frag_in_imag.x[j]};
            in_ele = cmul(in_ele, twiddle_factor);
            frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[j] = in_ele.y;
            twiddle_factor = cmul(twiddle_factor, twiddle_unit);
        }
        //**第一阶段（基16）结束**

        //**第二阶段（基16）开始**
        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        //**重点**
        // 之所以raw_row是这样计算，是因为需要从列优先转为行优先，故将列的计算结果赋予行，即raw_row
        int raw_row = threadIdx.x / 16 * 4 + threadIdx.x % 8 / 4 * 8 + threadIdx.x % 4;
        // 对于v100，fragment map的行（此处赋予raw_col)如下计算
        raw_col = threadIdx.x % 16 / 8 * 8;
        // 注意，fp32的fragment，每个只能存8个元素。
        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row;
            int col = j + raw_col;
            // 每个warp负责16*16的tile，所以tile的一行是16
            smem_in[warp_start + row * 16 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    // 注意，layer256与其他layer512和layer1024在此处有所不同
    // 之所以没有+
    //-输出，是因为，对于基256而言，采用双16一前一后合并，最后阶段是基16，并非基2，不必+
    //-输出.
    // 对基1024而言，分为三个阶段：16，16，4。所以最后，会针对基4结果进行加减输出
    __syncthreads();
    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        in[block_start + eid] = smem_in[eid];
    }
}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_256_0_A100(half2 *in, half *F_real, half *F_imag, int fft)
{
    const int map[8] = {0, 1, 4, 5, 2, 3, 6, 7};
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    LL block_start = (LL)blockIdx.x * 256 * (LL)CONT_SIZE;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_row = threadIdx.x % 4 * 2;
    int raw_col = threadIdx.x / 4;

    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 16 * 16)
    {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_sync;

        int warp_start = i + threadIdx.y * 256;

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            half2 ele = in[block_start + warp_start + row + col * 16];
            // half2 ele = smem_in[warp_start + row + col * 16]; // opt test
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        wmma::store_matrix_sync((half *)(smem_in + warp_start), frag_out_real, 16, wmma::mem_row_major);
        wmma::store_matrix_sync((half *)(smem_in + warp_start) + 256, frag_out_imag, 16, wmma::mem_row_major);

        /// @brief Redesign the data access mode without any load_matrix_sync by
        /// Aaron Chung(ltzhang@comp.hkbu.edu.hk). use a frag_sync to perform warp-level sync
        for (int i = 0; i < 8; i++)
        {
            frag_in_real.x[i] = frag_out_real.x[map[i]];
            frag_in_real.x[i + 8] = frag_out_real.x[map[i]];
            frag_in_imag.x[i] = frag_out_imag.x[map[i]];
            frag_in_imag.x[i + 8] = frag_out_imag.x[map[i]];
        }
        wmma::load_matrix_sync(frag_sync, (half *)(smem_in + warp_start), 16);

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            half2 in_ele = {frag_in_real.x[j], frag_in_imag.x[j]};
            in_ele = cmul(in_ele, W_N_K(256, row * col, fft));
            frag_in_real.x[8 + j] = frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = in_ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            in[block_start + warp_start + row * 16 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }
}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_256_1(int step, half2 *in, half *F_real, half *F_imag, int fft)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    LL block_start = (LL)blockIdx.y * (LL)step * 256 + (LL)blockIdx.x * (LL)CONT_SIZE;

    int b_c_col = threadIdx.x / 16 * 4 + threadIdx.x % 16 / 8 * 8 + threadIdx.x % 4;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    for (int i_start = 0; i_start < 256 * CONT_SIZE; i_start += NUM_WARP * 256)
    {
        int warp_start = i_start + threadIdx.y / 2 * 512 + threadIdx.y % 2 * 16;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;

        for (int j = 0; j < 16; ++j)
        {
            int col = b_c_col;
            int row = j;
            int eid = warp_start + row * 32 + col;
            half2 ele = in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE];
            frag_in_real.x[j] = ele.x;
            frag_in_imag.x[j] = ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        int acc_row = threadIdx.x / 16 * 4 + threadIdx.x % 8 / 4 * 8 + threadIdx.x % 4;
        int acc_col = threadIdx.x % 16 / 8 * 8;
        for (int j = 0; j < 8; ++j)
        {
            int row = acc_row;
            int col = j + acc_col;
            smem_in[warp_start + row * 32 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();

    for (int i_start = 0; i_start < CONT_SIZE / NUM_WARP; i_start++)
    {
        int warp_start = i_start * NUM_WARP * 16 + threadIdx.y * 16;
        int glb_col_2 = i_start * 4 + threadIdx.y / 2;
        half2 twiddle_unit_2 = W_N_K(256, glb_col_2, fft);
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;
        half2 twiddle_factor = {1.0, 0};
        for (int j = 0; j < 16; ++j)
        {
            int col = b_c_col;
            int row = j;
            half2 ele = smem_in[warp_start + row * 512 + col];
            ele = cmul(ele, twiddle_factor);
            frag_in_real.x[j] = ele.x;
            frag_in_imag.x[j] = ele.y;
            twiddle_factor = cmul(twiddle_factor, twiddle_unit_2);
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        int acc_row = threadIdx.x / 16 * 4 + threadIdx.x % 8 / 4 * 8 + threadIdx.x % 4;
        int acc_col = threadIdx.x % 16 / 8 * 8;
        for (int j = 0; j < 8; ++j)
        {
            int row = acc_row;
            int col = j + acc_col;
            smem_in[warp_start + row * 512 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();

    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE] = smem_in[eid];
    }
}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_256_1_A100(int step, half2 *in, half *F_real, half *F_imag, int fft)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    LL block_start = (LL)blockIdx.y * (LL)step * 256 + (LL)blockIdx.x * (LL)CONT_SIZE;

    int warp_col = blockIdx.x * CONT_SIZE + threadIdx.y % 2 * 16;

    int raw_row = threadIdx.x % 4 * 2;
    int raw_col = threadIdx.x / 4;

    half2 twiddle_factor;
    half2 twiddle_unit;

    for (int i = 0; i < 2; ++i)
    {
        int eid = i * 512 * 8 + threadIdx.y * 512 + threadIdx.x;
        for (int j = 0; j < 16; ++j)
        {
            smem_in[eid] = in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE];
            eid += 32;
        }
    }

    __syncthreads();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    half2 twiddle[8];

    for (int i_start = 0; i_start < 256 * CONT_SIZE; i_start += NUM_WARP * 256)
    {
        int warp_start = i_start + threadIdx.y / 2 * 512 + threadIdx.y % 2 * 16;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            int eid = warp_start + row * 32 + col;
            half2 ele = smem_in[eid];
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            smem_in[warp_start + row * 32 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();

    for (int i = 0; i < 2; ++i)
    {
        twiddle_unit = W_N_K(256, threadIdx.y + i * 8, fft);
        int eid = i * 32 * 8 + threadIdx.y * 32 + threadIdx.x;
        twiddle_factor = {1.0, 0};
        for (int j = 0; j < 16; ++j)
        {
            smem_in[eid] = cmul(smem_in[eid], twiddle_factor);
            eid += 512;
            twiddle_factor = cmul(twiddle_factor, twiddle_unit);
        }
    }

    __syncthreads();

    for (int i_start = 0; i_start < CONT_SIZE / NUM_WARP; i_start++)
    {
        int warp_start = i_start * NUM_WARP * 16 + threadIdx.y * 16;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            half2 ele = smem_in[warp_start + row * 512 + col];
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            smem_in[warp_start + row * 512 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();

    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE] = smem_in[eid];
    }
}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_512_0(half2 *in, half *F_real, half *F_imag, int fft)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    LL block_start = (LL)blockIdx.x * 512 * (LL)CONT_SIZE;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_col = threadIdx.x / 16 * 4 + threadIdx.x % 16 / 8 * 8 + threadIdx.x % 4;
    half2 twiddle_unit = W_N_K(256, raw_col, fft);
    half2 twiddle_two = W_N_K(512, t_block, fft);

    for (int i = 0; i < 512 * CONT_SIZE; i += NUM_WARP * 16 * 16)
    {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_tmp0;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_tmp1;

        int warp_start = i + threadIdx.y * 256;
        wmma::load_matrix_sync(frag_in_tmp0, (half *)(in + block_start + warp_start),
                               32); // stride = 32是因为16个实部和16个虚部在一行存储
        wmma::load_matrix_sync(frag_in_tmp1, (half *)(in + block_start + warp_start) + 16,
                               32); // 先16实部，再16虚部
        // for (int j = 0; j < 8; ++j) // 提升存取效率
        // {
        //     frag_in_imag.x[j] = (half *)(in + block_start + warp_start + 16)

        // }
        for (int j = 0; j < 8; ++j) // 提升存取效率
        {
            frag_in_real.x[j] = frag_in_tmp0.x[2 * j];
            frag_in_imag.x[j] = frag_in_tmp0.x[2 * j + 1];
            frag_in_real.x[8 + j] = frag_in_tmp1.x[2 * j];
            frag_in_imag.x[8 + j] = frag_in_tmp1.x[2 * j + 1];
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        wmma::store_matrix_sync((half *)(smem_in + warp_start), frag_out_real, 16, wmma::mem_row_major);
        wmma::store_matrix_sync((half *)(smem_in + warp_start) + 256, frag_out_imag, 16, wmma::mem_row_major);

        wmma::load_matrix_sync(frag_in_real, (half *)(smem_in + warp_start), 16);
        wmma::load_matrix_sync(frag_in_imag, (half *)(smem_in + warp_start) + 256, 16);

        half2 twiddle_factor = {1.0, 0};
        for (int j = 0; j < 16; ++j)
        {
            int row = j;
            int col = raw_col;
            half2 in_ele = {frag_in_real.x[j], frag_in_imag.x[j]};
            in_ele = cmul(in_ele, twiddle_factor);
            frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[j] = in_ele.y;
            twiddle_factor = cmul(twiddle_factor, twiddle_unit);
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        int raw_row = threadIdx.x / 16 * 4 + threadIdx.x % 8 / 4 * 8 + threadIdx.x % 4;
        raw_col = threadIdx.x % 16 / 8 * 8;
        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row;
            int col = j + raw_col;
            smem_in[warp_start + row * 16 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();
    for (int i = 0; i < 512 * CONT_SIZE; i += NUM_WARP * 32 * 2)
    {
        int eid = i + t_block;
        half2 ele_0 = smem_in[eid];
        half2 ele_1 = cmul(smem_in[eid + 256], twiddle_two);
        in[block_start + eid] = __hadd2(ele_0, ele_1);
        in[block_start + eid + 256] = __hsub2(ele_0, ele_1);
    }
}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_512_0_A100(half2 *in, half *F_real, half *F_imag, int fft)
{
    half tmp1 = in[0].x;
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    LL block_start = (LL)blockIdx.x * 512 * (LL)CONT_SIZE;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_row = threadIdx.x % 4 * 2;
    int raw_col = threadIdx.x / 4;
    half2 twiddle_two = W_N_K(512, t_block, fft);

    for (int i = 0; i < 512 * CONT_SIZE; i += NUM_WARP * 16 * 16)
    {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;

        int warp_start = i + threadIdx.y * 256;

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            half2 ele = in[block_start + warp_start + row + col * 16];
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        wmma::store_matrix_sync((half *)(smem_in + warp_start), frag_out_real, 16, wmma::mem_row_major);
        wmma::store_matrix_sync((half *)(smem_in + warp_start) + 256, frag_out_imag, 16, wmma::mem_row_major);

        wmma::load_matrix_sync(frag_in_real, (half *)(smem_in + warp_start), 16);
        wmma::load_matrix_sync(frag_in_imag, (half *)(smem_in + warp_start) + 256, 16);

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            half2 in_ele = {frag_in_real.x[j], frag_in_imag.x[j]};
            in_ele = cmul(in_ele, W_N_K(256, row * col, fft));
            frag_in_real.x[8 + j] = frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = in_ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            smem_in[warp_start + row * 16 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();
    for (int i = 0; i < 512 * CONT_SIZE; i += NUM_WARP * 32 * 2)
    {
        int eid = i + t_block;
        half2 ele_0 = smem_in[eid];
        half2 ele_1 = cmul(smem_in[eid + 256], twiddle_two);
        in[block_start + eid] = __hadd2(ele_0, ele_1);
        in[block_start + eid + 256] = __hsub2(ele_0, ele_1);
    }
}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_512_1(int step, half2 *in, half *F_real, half *F_imag, int fft)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    LL block_start = (LL)blockIdx.y * (LL)step * 512 + (LL)blockIdx.x * (LL)CONT_SIZE;

    int b_c_col = threadIdx.x / 16 * 4 + threadIdx.x % 16 / 8 * 8 + threadIdx.x % 4;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    for (int i_start = 0; i_start < 512 * CONT_SIZE; i_start += NUM_WARP * 256)
    {
        int warp_start = i_start + threadIdx.y * 256;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;
        for (int j = 0; j < 16; ++j)
        {
            int col = b_c_col;
            int row = j;
            int eid = warp_start + row * CONT_SIZE + col;
            half2 ele = in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE];
            frag_in_real.x[j] = ele.x;
            frag_in_imag.x[j] = ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        int acc_row = threadIdx.x / 16 * 4 + threadIdx.x % 8 / 4 * 8 + threadIdx.x % 4;
        int acc_col = threadIdx.x % 16 / 8 * 8;
        for (int j = 0; j < 8; ++j)
        {
            int row = acc_row;
            int col = j + acc_col;
            smem_in[warp_start + row * CONT_SIZE + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();

    for (int i_start = 0; i_start < 4; i_start++)
    {
        int warp_start = i_start % 2 * NUM_WARP * 16 + i_start / 2 * 256 * CONT_SIZE + threadIdx.y * 16;
        int glb_col_2 = i_start % 2 * 8 + threadIdx.y;
        float2 twiddle_unit_2 = W_N_K_fp32(256, glb_col_2, fft); // mixed precision improved
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;
        float2 twiddle_factor = {1.0, 0}; // mixed precision improved
        for (int j = 0; j < 16; ++j)
        {
            int col = b_c_col;
            int row = j;
            half2 ele = smem_in[warp_start + row * 256 + col];
            ele = cmul_mixed(ele, twiddle_factor); // mixed precision improved
            frag_in_real.x[j] = ele.x;
            frag_in_imag.x[j] = ele.y;
            twiddle_factor = cuCmulf(twiddle_factor, twiddle_unit_2); // mixed precision improved
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        int acc_row = threadIdx.x / 16 * 4 + threadIdx.x % 8 / 4 * 8 + threadIdx.x % 4;
        int acc_col = threadIdx.x % 16 / 8 * 8;
        for (int j = 0; j < 8; ++j)
        {
            int row = acc_row;
            int col = j + acc_col;
            smem_in[warp_start + row * 256 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();

    float2 twiddle_unit_2 = W_N_K_fp32(512, 256 / CONT_SIZE, fft);     // mixed precision improved
    float2 twiddle_factor = W_N_K_fp32(512, t_block / CONT_SIZE, fft); // mixed precision improved
    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        half2 ele_0 = smem_in[eid];
        half2 ele_1 = cmul_mixed(smem_in[eid + 256 * CONT_SIZE],
                                 twiddle_factor); // mixed precision improved
        in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE] = __hadd2(ele_0, ele_1);
        eid += 256 * CONT_SIZE;
        in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE] = __hsub2(ele_0, ele_1);
        twiddle_factor = cuCmulf(twiddle_factor, twiddle_unit_2); // mixed precision improved
    }
}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_512_1_A100(int step, half2 *in, half *F_real, half *F_imag, int fft)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    LL block_start = (LL)blockIdx.y * (LL)step * 512 + (LL)blockIdx.x * (LL)CONT_SIZE;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_row = threadIdx.x % 4 * 2;
    int raw_col = threadIdx.x / 4;

    half2 twiddle_factor;
    half2 twiddle_unit;

    for (int i = 0; i < 2; ++i)
    {
        int eid = i * 512 * 8 + threadIdx.y * 512 + threadIdx.x / 16 * 256 + threadIdx.x % 16;
        for (int j = 0; j < 16; ++j)
        {
            smem_in[eid] = in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE];
            eid += 16;
        }
    }

    __syncthreads();

    for (int i_start = 0; i_start < 512 * CONT_SIZE; i_start += NUM_WARP * 256)
    {
        int warp_start = i_start + threadIdx.y * 256;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            int eid = warp_start + row * 16 + col;
            half2 ele = smem_in[eid];
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            smem_in[warp_start + row * 16 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();

    for (int i = 0; i < 2; ++i)
    {
        twiddle_unit = W_N_K(256, threadIdx.y * 2 + threadIdx.x / 16, fft);
        int eid = i * 16 * 16 * 16 + threadIdx.y * 32 + threadIdx.x;
        twiddle_factor = {1.0, 0};
        for (int j = 0; j < 16; ++j)
        {
            smem_in[eid] = cmul(smem_in[eid], twiddle_factor);
            eid += 256;
            twiddle_factor = cmul(twiddle_factor, twiddle_unit);
        }
    }

    __syncthreads();

    for (int i_start = 0; i_start < 4; i_start++)
    {
        int warp_start = i_start % 2 * NUM_WARP * 16 + i_start / 2 * 256 * CONT_SIZE + threadIdx.y * 16;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            half2 ele = smem_in[warp_start + row * 256 + col];
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            smem_in[warp_start + row * 256 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();

    half2 twiddle_unit_2 = W_N_K(512, 256 / CONT_SIZE, fft);
    twiddle_factor = W_N_K(512, t_block / CONT_SIZE, fft);
    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        half2 ele_0 = smem_in[eid];
        half2 ele_1 = cmul(smem_in[eid + 256 * CONT_SIZE], twiddle_factor);
        in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE] = __hadd2(ele_0, ele_1);
        eid += 256 * CONT_SIZE;
        in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE] = __hsub2(ele_0, ele_1);
        twiddle_factor = cmul(twiddle_factor, twiddle_unit_2);
    }
}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_1024_0(half2 *in, half *F_real, half *F_imag, int fft)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    LL block_start = (LL)blockIdx.x * 1024 * (LL)CONT_SIZE;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_col = threadIdx.x / 16 * 4 + threadIdx.x % 16 / 8 * 8 + threadIdx.x % 4;
    half2 twiddle_unit = W_N_K(256, raw_col, fft);

    for (int i = 0; i < 1024 * CONT_SIZE; i += NUM_WARP * 16 * 16)
    {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_tmp0;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_tmp1;

        int warp_start = i + threadIdx.y * 256;
        wmma::load_matrix_sync(frag_in_tmp0, (half *)(in + block_start + warp_start), 32);
        wmma::load_matrix_sync(frag_in_tmp1, (half *)(in + block_start + warp_start) + 16, 32);

        for (int j = 0; j < 8; ++j)
        {
            frag_in_real.x[j] = frag_in_tmp0.x[2 * j];
            frag_in_imag.x[j] = frag_in_tmp0.x[2 * j + 1];
            frag_in_real.x[8 + j] = frag_in_tmp1.x[2 * j];
            frag_in_imag.x[8 + j] = frag_in_tmp1.x[2 * j + 1];
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        wmma::store_matrix_sync((half *)(smem_in + warp_start), frag_out_real, 16, wmma::mem_row_major);
        wmma::store_matrix_sync((half *)(smem_in + warp_start) + 256, frag_out_imag, 16, wmma::mem_row_major);

        wmma::load_matrix_sync(frag_in_real, (half *)(smem_in + warp_start), 16);
        wmma::load_matrix_sync(frag_in_imag, (half *)(smem_in + warp_start) + 256, 16);

        half2 twiddle_factor = {1.0, 0};
        for (int j = 0; j < 16; ++j)
        {
            int row = j;
            int col = raw_col;
            half2 in_ele = {frag_in_real.x[j], frag_in_imag.x[j]};
            in_ele = cmul(in_ele, twiddle_factor);
            frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[j] = in_ele.y;
            twiddle_factor = cmul(twiddle_factor, twiddle_unit);
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        int raw_row = threadIdx.x / 16 * 4 + threadIdx.x % 8 / 4 * 8 + threadIdx.x % 4;
        raw_col = threadIdx.x % 16 / 8 * 8;
        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row;
            int col = j + raw_col;
            smem_in[warp_start + row * 16 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    half2 twiddle_1024_1 = W_N_K(1024, t_block, fft);
    half2 twiddle_1024_2 = cmul(twiddle_1024_1, twiddle_1024_1);
    half2 twiddle_1024_3 = cmul(twiddle_1024_2, twiddle_1024_1);

    __syncthreads();
    for (int i = 0; i < 1024 * CONT_SIZE; i += NUM_WARP * 32 * 4)
    {
        int eid = i + t_block;
        half2 ele0 = smem_in[eid];
        half2 ele1 = cmul(smem_in[eid + 256], twiddle_1024_1);
        half2 ele2 = cmul(smem_in[eid + 512], twiddle_1024_2);
        half2 ele3 = cmul(smem_in[eid + 768], twiddle_1024_3);
        in[block_start + eid] = ele0 + ele1 + ele2 + ele3;
        in[block_start + eid + 256] = ele0 + half2({ele1.y, -ele1.x}) - ele2 + half2({-ele3.y, ele3.x});
        in[block_start + eid + 512] = ele0 - ele1 + ele2 - ele3;
        in[block_start + eid + 768] = ele0 + half2({-ele1.y, ele1.x}) - ele2 + half2({ele3.y, -ele3.x});
    }
}

template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_1024_0_A100(half2 *in, half *F_real, half *F_imag, int fft)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 1024 * CONT_SIZE;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_row = threadIdx.x % 4 * 2;
    int raw_col = threadIdx.x / 4;

    for (int i = 0; i < 1024 * CONT_SIZE; i += NUM_WARP * 16 * 16)
    {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;

        int warp_start = i + threadIdx.y * 256;

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            half2 ele = in[block_start + warp_start + row + col * 16];
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        wmma::store_matrix_sync((half *)(smem_in + warp_start), frag_out_real, 16, wmma::mem_row_major);
        wmma::store_matrix_sync((half *)(smem_in + warp_start) + 256, frag_out_imag, 16, wmma::mem_row_major);

        wmma::load_matrix_sync(frag_in_real, (half *)(smem_in + warp_start), 16);
        wmma::load_matrix_sync(frag_in_imag, (half *)(smem_in + warp_start) + 256, 16);

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            half2 in_ele = {frag_in_real.x[j], frag_in_imag.x[j]};
            in_ele = cmul(in_ele, W_N_K(256, row * col, fft));
            frag_in_real.x[8 + j] = frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = in_ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            smem_in[warp_start + row * 16 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    half2 twiddle_1024_1 = W_N_K(1024, t_block, fft);
    half2 twiddle_1024_2 = cmul(twiddle_1024_1, twiddle_1024_1);
    half2 twiddle_1024_3 = cmul(twiddle_1024_2, twiddle_1024_1);

    __syncthreads();
    for (int i = 0; i < 1024 * CONT_SIZE; i += NUM_WARP * 32 * 4)
    {
        int eid = i + t_block;
        half2 ele0 = smem_in[eid];
        half2 ele1 = cmul(smem_in[eid + 256], twiddle_1024_1);
        half2 ele2 = cmul(smem_in[eid + 512], twiddle_1024_2);
        half2 ele3 = cmul(smem_in[eid + 768], twiddle_1024_3);
        in[block_start + eid] = ele0 + ele1 + ele2 + ele3;
        in[block_start + eid + 256] = ele0 + half2({ele1.y, -ele1.x}) - ele2 + half2({-ele3.y, ele3.x});
        in[block_start + eid + 512] = ele0 - ele1 + ele2 - ele3;
        in[block_start + eid + 768] = ele0 + half2({-ele1.y, ele1.x}) - ele2 + half2({ele3.y, -ele3.x});
    }
}

//<<<(F,H,1),(W,1,1)>>>
__global__ void fft_cpx_point_wise_mul_add(half *signal, half *filter, float *result_float, int C, int F, int H, int W)
{
    const LL which_filter = blockIdx.x;
    const LL which_line_in_tile = blockIdx.y;
    const LL which_ele_in_line = threadIdx.x;
    const LL tile_size = H * W;

    LL idx_in_signal = which_line_in_tile * W + which_ele_in_line;
    LL idx_in_filter = which_filter * tile_size * C + which_line_in_tile * W + which_ele_in_line;
    LL idx_in_result = which_filter * tile_size + which_line_in_tile * W + which_ele_in_line;

    float2 tmp, a_ele, b_ele;
    for (int i = 0; i < C; i++)
    {
        a_ele.x = __half2float(signal[2 * (idx_in_signal + i * tile_size)]);
        a_ele.y = __half2float(signal[2 * (idx_in_signal + i * tile_size) + 1]);
        b_ele.x = __half2float(filter[2 * (idx_in_filter + i * tile_size)]);
        b_ele.y = __half2float(filter[2 * (idx_in_filter + i * tile_size) + 1]);
        tmp = cmul_full(a_ele, b_ele);
        result_float[2 * idx_in_result] += tmp.x;
        result_float[2 * idx_in_result + 1] += tmp.y;
    }
}

//<<<(F,1,1),(H,1,1)>>>
__global__ void fft_cpx_point_wise_quant_p1(float *result_float, float *max_per_line, int H, int W)
{
    const int which_result = blockIdx.x;
    const int which_line_in_tile = threadIdx.x;
    const int which_line_in_global = which_result * H + threadIdx.x;
    const int tile_size = H * W;

    const int idx_start = which_result * tile_size + which_line_in_tile * W;
    float max = -19777773;
    for (int i = 0; i < W; i++)
    {
        max = abs(result_float[2 * (idx_start + i)]) > max ? abs(result_float[2 * (idx_start + i)]) : max;
        max = abs(result_float[2 * (idx_start + i) + 1]) > max ? abs(result_float[2 * (idx_start + i) + 1]) : max;
    }
    max_per_line[which_line_in_global] = max;
}
//<<<(1,1,1),(F,1,1)>>>
__global__ void fft_cpx_point_wise_quant_p2(float *max_per_line, float *max_per_tile, int H, int W)
{
    const int which_result = threadIdx.x;

    const int idx_start = which_result * H;
    float max = -19777773;

    for (int i = 0; i < H; i++)
    {
        max = abs(max_per_line[idx_start + i]) > max ? abs(max_per_line[idx_start + i]) : max;
    }
    max_per_tile[which_result] = max <= 1000 ? 1 : pow(10, (int)log10(max) - 1);
}
//<<<(F,H,1),(W,2,1)>>>
__global__ void fft_cpx_point_wise_quant_p3(float *result_float, half *result_half, float *max_per_tile, int *rev_x, int *rev_y, int H, int W)
{
    const LL which_result = blockIdx.x;
    const LL which_line_in_tile = blockIdx.y;
    const LL which_ele_in_line = threadIdx.x;
    const LL idx = which_result * H * W + which_line_in_tile * W + which_ele_in_line;
    result_half[2 * idx + 0] = __float2half(result_float[2 * (which_result * H * W + rev_y[which_line_in_tile] * W + rev_x[which_ele_in_line]) + 0] / max_per_tile[which_result]);
    result_half[2 * idx + 1] = __float2half(result_float[2 * (which_result * H * W + rev_y[which_line_in_tile] * W + rev_x[which_ele_in_line]) + 1] / max_per_tile[which_result]);
}

/*Pointwise Mul with quant after fft(signal) and fft(filter)*/
void fft_conv2d_half::fft_pointwise(float scale, bool use_more_G_memory)
{
    int H = padded_y;
    int W = padded_x;

    dim3 threadsPerblock1(W, 1, 1);
    dim3 blocks1(F, H, 1);
    fft_cpx_point_wise_mul_add<<<blocks1, threadsPerblock1>>>(cu_signal_padded, cu_filter_padded, cu_pointwise_float, S, F, H, W);
    cudaDeviceSynchronize();
    cudaFree(cu_signal_padded);
    cudaFree(cu_filter_padded);

    dim3 threadsPerblock2(H, 1, 1);
    dim3 blocks2(F, 1, 1);
    fft_cpx_point_wise_quant_p1<<<blocks2, threadsPerblock2>>>(cu_pointwise_float, max_per_line, H, W);
    cudaDeviceSynchronize();

    dim3 threadsPerblock3(F, 1, 1);
    dim3 blocks3(1, 1, 1);
    fft_cpx_point_wise_quant_p2<<<blocks3, threadsPerblock3>>>(max_per_line, max_per_tile, H, W);
    cudaDeviceSynchronize();

    dim3 threadsPerblock4(W, 1, 1);
    dim3 blocks4(F, H, 1);
    fft_cpx_point_wise_quant_p3<<<blocks4, threadsPerblock4>>>(cu_pointwise_float, cu_pointwise_half, max_per_tile, rev_x_cu, rev_y_cu, H, W);
    cudaDeviceSynchronize();
}

//<<<(F,H,1),(W,2,1)>>>
__global__ void fft_cpx_point_wise_dequant(half *result_half, double *result_double, float *max_per_tile, int H, int W)
{
    const LL which_result = blockIdx.x;
    const LL which_line_in_tile = blockIdx.y;
    const LL which_ele_in_line = threadIdx.x;
    const LL idx = which_result * H * W + which_line_in_tile * W + which_ele_in_line;

    result_double[2 * idx + 0] = (double)__half2float(result_half[2 * idx + 0]) * (double)max_per_tile[which_result];
    result_double[2 * idx + 1] = (double)__half2float(result_half[2 * idx + 1]) * (double)max_per_tile[which_result];
}
/*De-quant after ifft(pointwise)*/
void fft_conv2d_half::fft_conv_final_stage()
{
    int H = padded_y;
    int W = padded_x;

    dim3 threadsPerblock5(W, 1, 1);
    dim3 blocks5(F, H, 1);
    fft_cpx_point_wise_dequant<<<blocks5, threadsPerblock5>>>(cu_pointwise_half, cu_conv2d_forward_result_padded, max_per_tile, H, W);
    cudaDeviceSynchronize();
}

//<<<(F,S,H),(W,1,1)>>>
__global__ void pointwise_for_delta_weight_kernel(half *signal, half *delta_in, half *result_half, int C, int F, int H, int W)
{
    const LL which_filter = blockIdx.x;
    const LL which_signal = blockIdx.y;

    const LL which_line_in_tile = blockIdx.z;
    const LL which_ele_in_line = threadIdx.x;
    const LL tile_size = H * W;

    LL idx_in_signal = which_signal * tile_size + which_line_in_tile * W + which_ele_in_line;
    LL idx_in_filter = which_filter * tile_size + which_line_in_tile * W + which_ele_in_line;
    LL idx_in_result = which_filter * C * tile_size + which_signal * tile_size + which_line_in_tile * W + which_ele_in_line;

    float2 tmp, a_ele, b_ele;
    a_ele.x = __half2float(signal[2 * idx_in_signal]);
    a_ele.y = __half2float(signal[2 * idx_in_signal + 1]);
    b_ele.x = __half2float(delta_in[2 * idx_in_filter]);
    b_ele.y = __half2float(delta_in[2 * idx_in_filter + 1]);
    tmp = cmul_full(a_ele, b_ele);
    result_half[2 * idx_in_result] = __float2half(tmp.x);
    result_half[2 * idx_in_result + 1] = __float2half(tmp.y);
}

/*BackPropagation-Pointwise mul of d(filter), after fft(signal) and fft(delta_in)*/
void fft_conv2d_half::pointwise_for_delta_weight(int padded_x)
{
    int H = padded_x;
    int W = padded_x;

    dim3 threadsPerblock1(W, 1, 1);
    dim3 blocks1(F, S, H);
    pointwise_for_delta_weight_kernel<<<blocks1, threadsPerblock1>>>(cu_signal_padded, cu_delta_in_padded, cu_pointwise_for_delta_filter, S, F, H, W);
    cudaDeviceSynchronize();
}

//<<<(S,H,1),(W,1,1)>>>
__global__ void pointwise_for_delta_input_kernel(half *delta_in, half *filter, half *result_half, int C, int F, int H, int W)
{
    const LL which_result = blockIdx.x;
    const LL which_line_in_tile = blockIdx.y;
    const LL which_ele_in_line = threadIdx.x;
    const LL tile_size = H * W;
    LL idx_in_result = which_result * tile_size + which_line_in_tile * W + which_ele_in_line;
    half2 tmp, a_ele, b_ele;

    for (int i = 0; i < F; i++)
    {
        a_ele.x = delta_in[(i * tile_size + which_line_in_tile * W + which_ele_in_line) * 2];
        a_ele.y = delta_in[(i * tile_size + which_line_in_tile * W + which_ele_in_line) * 2 + 1];
        b_ele.x = filter[(i * C * tile_size + which_result * tile_size + which_line_in_tile * W + which_ele_in_line) * 2];
        b_ele.y = filter[(i * C * tile_size + which_result * tile_size + which_line_in_tile * W + which_ele_in_line) * 2 + 1];
        tmp = cmul(a_ele, b_ele);
        result_half[2 * idx_in_result] += tmp.x;
        result_half[2 * idx_in_result + 1] += tmp.y;
    }
}

/*BackPropagation-Pointwise mul of d(input), after fft(filter) and fft(delta_in)*/
void fft_conv2d_half::pointwise_for_delta_input(int padded_x, half *delta_for_input_pointwise)
{
    int H = padded_x;
    int W = padded_x;

    dim3 threadsPerblock1(W, 1, 1);
    dim3 blocks1(S, H, 1);
    pointwise_for_delta_input_kernel<<<blocks1, threadsPerblock1>>>(cu_delta_in_padded, cu_filter_padded, delta_for_input_pointwise, S, F, H, W);
    cudaDeviceSynchronize();
}

__global__ void fft_conv2d_bit_rev_real_pad_to_cpx_double2half(double *src, half *dst, int *rev_x, int *rev_y, int dx, int dy, int x, int y)
{
    const LL batch = blockIdx.x;
    const LL line = blockIdx.y;
    const LL ele = threadIdx.x;
    const LL idx_in_dst = batch * dx * dy + line * dx + ele;
    if (rev_y[line] < y && rev_x[ele] < x)
    {
        dst[2 * idx_in_dst] = __double2half(src[batch * x * y + rev_y[line] * x + rev_x[ele]]);
    }
}

__global__ void fft_conv2d_inverse_filter(double *src, int x, int y)
{
    const LL batch = blockIdx.x;
    const LL line = blockIdx.y;
    const LL ele = threadIdx.x;

    const LL idx_in_src_1 = batch * x * y + line * x + ele;
    const LL idx_in_src_2 = batch * x * y + swapnum_line(y, line) * x + swapnum_ele(x, ele);
    swap(src[idx_in_src_2], src[idx_in_src_1]);
}
__global__ void fft_conv2d_inverse_filter_odd(double *src, int x, int y)
{
    const LL batch = blockIdx.x;
    const LL line = y / 2;
    const LL ele = threadIdx.x;
    const LL idx_in_src_1 = batch * x * y + line * x + ele;
    const LL idx_in_src_2 = batch * x * y + line * x + x - ele - 1;

    swap(src[idx_in_src_2], src[idx_in_src_1]);
}
/*Reverse in-place and pad to CPX matrix of signal. Rotate, reverse in-place and pad to CPX matrix of filter. Before fft(signal) and fft(filter)*/
void fft_conv2d_half::fft_conv2d_inv_rev_pad()
{
    dim3 threadsPerblock1(fx, 1, 1);
    dim3 blocks1(F * S, (int)(fy / 2), 1);
    fft_conv2d_inverse_filter<<<blocks1, threadsPerblock1>>>(cu_filter, fx, fy);
    if (fy & 1)
    {
        dim3 threadsPerblock2(fx / 2, 1, 1);
        dim3 blocks2(F * S, 1, 1);
        fft_conv2d_inverse_filter_odd<<<blocks2, threadsPerblock2>>>(cu_filter, fx, fy);
    }
    cudaDeviceSynchronize();

    dim3 threadsPerblock_padsignal(padded_x, 1, 1);
    dim3 blocks_padsignal(S, padded_y, 1);
    fft_conv2d_bit_rev_real_pad_to_cpx_double2half<<<blocks_padsignal, threadsPerblock_padsignal>>>(cu_signal, cu_signal_padded, rev_x_cu, rev_y_cu, padded_x, padded_y, sx, sy);

    dim3 threadsPerblock_pad_filter(padded_x, 1, 1);
    dim3 blocks_pad_filter(F * S, padded_y, 1);
    fft_conv2d_bit_rev_real_pad_to_cpx_double2half<<<blocks_pad_filter, threadsPerblock_pad_filter>>>(cu_filter, cu_filter_padded, rev_x_cu, rev_y_cu, padded_x, padded_y, fx, fy);
    cudaDeviceSynchronize();
}
__global__ void rev_kernel(double *src, half *dst, int *rev_x_cu, int *rev_y_cu, int x, int y)
{
    const LL which_tile = blockIdx.x;
    const LL which_line_in_tile = blockIdx.y;
    const LL which_ele_in_line = threadIdx.x;
    int tile_size = x * y;
    dst[2 * (which_tile * tile_size + which_line_in_tile * x + which_ele_in_line)] = __float2half(src[2 * (which_tile * tile_size + rev_y_cu[which_line_in_tile] * x + rev_x_cu[which_ele_in_line])]);
    dst[2 * (which_tile * tile_size + which_line_in_tile * x + which_ele_in_line) + 1] = __float2half(src[2 * (which_tile * tile_size + rev_y_cu[which_line_in_tile] * x + rev_x_cu[which_ele_in_line]) + 1]);
}
/*Reverse in-place CPX Matrix*/
void fft_conv2d_half::rev(double *src, half *dst, int x, int y, int batch, int offset, cudaStream_t stream)
{
    int *rev_x, *rev_y;
    cudaMalloc(&rev_x, sizeof(int) * 1024);
    cudaMalloc(&rev_y, sizeof(int) * 1024);

    if (x == 256)
    {
        cudaMemcpy(rev_x, rev_256, sizeof(int) * 1024, cudaMemcpyHostToDevice);
        cudaMemcpy(rev_y, rev_256, sizeof(int) * 1024, cudaMemcpyHostToDevice);
    }
    else if (x == 512)
    {
        cudaMemcpy(rev_x, rev_512, sizeof(int) * 1024, cudaMemcpyHostToDevice);
        cudaMemcpy(rev_y, rev_512, sizeof(int) * 1024, cudaMemcpyHostToDevice);
    }
    dim3 threadsPerblock_padsignal(x, 1, 1);
    dim3 blocks_padsignal(batch, y, 1);
    rev_kernel<<<blocks_padsignal, threadsPerblock_padsignal, 0, stream>>>(src, dst, rev_x, rev_y, x, y);
    // cudaDeviceSynchronize();

    // TODO: delete or keep?
    cudaFree(rev_x);
    cudaFree(rev_y);
}

/*BackPropagation-Reverse in-place and pad to CPX matrix of signal. Rotate, reverse in-place and pad to CPX matrix of cu_delta_in. Before fft(signal) and fft(cu_delta_in)*/
void fft_conv2d_half::inv_pad_deltain_signal(int padded_x)
{
    dim3 threadsPerblock_padsignal(padded_x, 1, 1);
    dim3 blocks_padsignal(S, padded_y, 1);
    fft_conv2d_bit_rev_real_pad_to_cpx_double2half<<<blocks_padsignal, threadsPerblock_padsignal>>>(cu_signal, cu_signal_padded, rev_x_cu, rev_y_cu, padded_x, padded_y, sx, sy);

    dim3 threadsPerblock1(output_x, 1, 1);
    dim3 blocks1(F, (int)(output_y / 2), 1);
    fft_conv2d_inverse_filter<<<blocks1, threadsPerblock1>>>(cu_delta_in, output_x, output_y);
    if (output_y & 1)
    {
        dim3 threadsPerblock2(output_x / 2, 1, 1);
        dim3 blocks2(F, 1, 1);
        fft_conv2d_inverse_filter_odd<<<blocks2, threadsPerblock2>>>(cu_delta_in, output_x, output_y);
    }
    cudaDeviceSynchronize();

    dim3 threadsPerblock_pad_filter(padded_x, 1, 1);
    dim3 blocks_pad_filter(F, padded_x, 1);
    fft_conv2d_bit_rev_real_pad_to_cpx_double2half<<<blocks_pad_filter, threadsPerblock_pad_filter>>>(cu_delta_in, cu_delta_in_padded, rev_x_cu, rev_y_cu, padded_x, padded_x, output_x, output_y);
    cudaDeviceSynchronize();
}
__global__ void fft_conv2d_rev_pad_kernel(double *cu_delta_in, double *cu_delta_in_0_padded, int x, int y, int dst_x, int dst_y, int k)
{
    const LL idx_in_src = (LL)blockIdx.x * x * y + (LL)blockIdx.y * x + threadIdx.x;
    const LL idx_in_dst = (LL)blockIdx.x * dst_x * dst_y + (LL)(blockIdx.y + k) * dst_x + threadIdx.x + k;
    cu_delta_in_0_padded[idx_in_dst] = cu_delta_in[idx_in_src];
}
/*Reverse in-place and pad to CPX Matrix of signal and filter*/
void fft_conv2d_half::fft_conv2d_rev_pad(double *siganl, double *filter, half *signal_padded, half *filter_padded, int padded_x, int sx, int fx, int S, int F)
{
    dim3 threadsPerblock_padsignal(padded_x, 1, 1);
    dim3 blocks_padsignal(S, padded_x, 1);
    fft_conv2d_bit_rev_real_pad_to_cpx_double2half<<<blocks_padsignal, threadsPerblock_padsignal>>>(siganl, signal_padded, rev_x_cu, rev_y_cu, padded_x, padded_x, sx, sx);

    dim3 threadsPerblock_pad_filter(padded_x, 1, 1);
    dim3 blocks_pad_filter(F * S, padded_x, 1);
    fft_conv2d_bit_rev_real_pad_to_cpx_double2half<<<blocks_pad_filter, threadsPerblock_pad_filter>>>(filter, filter_padded, rev_x_cu, rev_y_cu, padded_x, padded_x, fx, fx);
    cudaDeviceSynchronize();
}

__global__ void pad_n_1_circles_of_0_kernel(double *cu_delta_in, double *cu_delta_in_0_padded, int x, int y, int dst_x, int dst_y, int k)
{
    const LL idx_in_src = (LL)blockIdx.x * x * y + (LL)blockIdx.y * x + threadIdx.x;
    const LL idx_in_dst = (LL)blockIdx.x * dst_x * dst_y + (LL)(blockIdx.y + k) * dst_x + threadIdx.x + k;
    cu_delta_in_0_padded[idx_in_dst] = cu_delta_in[idx_in_src];
}
/*BackPropagation-Pad {kernel_x-1} circles of 0 to delta_in. Here, k = kernel_x-1*/
void fft_conv2d_half::pad_fx_1_circles_of_0(double *cu_delta_in_0_padded, int k, int x, int y)
{
    // double *cu_delta_in,cu_delta_in_0_padded
    int dst_x = x + 2 * k;
    int dst_y = y + 2 * k;
    dim3 threadsPerblock1(x, 1, 1);
    dim3 blocks1(F, y, 1);
    pad_n_1_circles_of_0_kernel<<<threadsPerblock1, blocks1>>>(cu_delta_in, cu_delta_in_0_padded, x, y, dst_x, dst_y, k);
    cudaDeviceSynchronize();
}

__global__ void read_out_scale_half2double_kernel(half *src, double *dst, int src_x, int src_y, int dst_x, int dst_y, int fx, double scale)
{
    int tile_size_src = src_x * src_y;
    int tile_size_dst = dst_x * dst_y;
    LL idx_in_src = blockIdx.x * tile_size_src + (blockIdx.y + fx - 1) * src_x + threadIdx.x + fx - 1;
    LL idx_in_dst = blockIdx.x * tile_size_dst + blockIdx.y * dst_x + threadIdx.x;
    dst[idx_in_dst] = __half2float(src[idx_in_src * 2]) * scale;
}
/*Read half CPX matrix to double REAL matrix with scale, to get correct fft_conv2d result*/
void fft_conv2d_half::read_out_scale_half2double(half *src, double *dst, double *dst_in_host, int src_x, int src_y, int dst_x, int dst_y, int batch, double scale)
{
    dim3 threadsPerblock1(dst_x, 1, 1);
    dim3 blocks1(batch, dst_y, 1);
    read_out_scale_half2double_kernel<<<blocks1, threadsPerblock1>>>(src, dst, src_x, src_y, dst_x, dst_y, fx, scale);
    cudaDeviceSynchronize();
    cudaMemcpy(dst_in_host, dst, sizeof(double) * dst_x * dst_y * batch, cudaMemcpyDeviceToHost);
}
__global__ void read_out_scale_double2double_kernel(double *src, double *dst, int src_x, int src_y, int dst_x, int dst_y, int fx, double scale)
{
    int tile_size_src = src_x * src_y;
    int tile_size_dst = dst_x * dst_y;
    LL idx_in_src = blockIdx.x * tile_size_src + (blockIdx.y + fx - 1) * src_x + threadIdx.x + fx - 1;
    LL idx_in_dst = blockIdx.x * tile_size_dst + blockIdx.y * dst_x + threadIdx.x;
    dst[idx_in_dst] = src[idx_in_src * 2] * scale;
}
/*Read double CPX matrix to double REAL matrix with scale, to get correct fft_conv2d result*/
void fft_conv2d_half::read_out_scale_double2double(double *src, double *dst, double *dst_in_host, int src_x, int src_y, int dst_x, int dst_y, int batch, double scale)
{
    dim3 threadsPerblock1(dst_x, 1, 1);
    dim3 blocks1(batch, dst_y, 1);
    read_out_scale_double2double_kernel<<<blocks1, threadsPerblock1>>>(src, dst, src_x, src_y, dst_x, dst_y, fx, scale);
    cudaDeviceSynchronize();
    cudaMemcpy(dst_in_host, dst, sizeof(double) * dst_x * dst_y * batch, cudaMemcpyDeviceToHost);
}
__global__ void half2double_kernel(half *src, double *dst, int x, int y, int item_per_group)
{
    LL idx = blockIdx.x * x * y + blockIdx.y * x + threadIdx.x;
    for (int i = 0; i < item_per_group; i++)
    {
        dst[idx * item_per_group + i] = __half2float(src[idx * item_per_group + i]);
    }
}
/*Transform half matrix to double matrix in group mode on GPU*/
void fft_conv2d_half::half2double(half *src, double *dst, int x, int y, int item_per_group, int batch, cudaStream_t stream)
{
    dim3 threadsPerblock1(x, 1, 1);
    dim3 blocks1(batch, y, 1);
    half2double_kernel<<<blocks1, threadsPerblock1, 0, stream>>>(src, dst, x, y, item_per_group);
}
void fft_conv2d_half::print_cu(double *src, int x, int y, int item_per_group, int batch)
{
    printf("\n-------------\n");
    double *in_host_src = (double *)malloc(sizeof(double) * x * y * item_per_group * batch);
    cudaMemcpy(in_host_src, src, sizeof(double) * x * y * item_per_group * batch, cudaMemcpyDeviceToHost);
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < y; i++)
        {
            printf("%d\n", i);
            for (int j = 0; j < x; j++)
            {
                for (int k = 0; k < item_per_group; k++)
                {
                    printf("%f", in_host_src[item_per_group * (b * x * y + i * x + j) + k]);
                    if (k < item_per_group - 1) printf("-");
                }
                printf(" ");
            }
            printf("\n");
        }
        printf("-------------\n");
    }
}
void fft_conv2d_half::print_cu(float *src, int x, int y, int item_per_group, int batch)
{
    printf("\n-------------\n");
    float *in_host_src = (float *)malloc(sizeof(float) * x * y * item_per_group * batch);
    cudaMemcpy(in_host_src, src, sizeof(float) * x * y * item_per_group * batch, cudaMemcpyDeviceToHost);
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < y; i++)
        {
            printf("%d\n", i);
            for (int j = 0; j < x; j++)
            {
                for (int k = 0; k < item_per_group; k++)
                {
                    printf("%f", in_host_src[item_per_group * (b * x * y + i * x + j) + k]);
                    if (k < item_per_group - 1) printf("-");
                }
                printf(" ");
            }
            printf("\n");
        }
        printf("-------------\n");
    }
}

void fft_conv2d_half::print_cu(half *src, int x, int y, int item_per_group, int batch)
{
    printf("\n-------------\n");
    half *in_host_src = (half *)malloc(sizeof(half) * x * y * item_per_group * batch);
    cudaMemcpy(in_host_src, src, sizeof(half) * x * y * item_per_group * batch, cudaMemcpyDeviceToHost);
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < y; i++)
        {
            printf("%d\n", i);
            for (int j = 0; j < x; j++)
            {
                for (int k = 0; k < item_per_group; k++)
                {
                    printf("%f", __half2float(in_host_src[item_per_group * (b * x * y + i * x + j) + k]));
                    if (k < item_per_group - 1) printf("-");
                }
                printf(" ");
            }
            printf("\n");
        }
        printf("-------------\n");
    }
}
void fft_conv2d_half::TensorCoreFFTExec(fftPlan plan, half *data, cudaStream_t stream)
{
    // fft = 1 -> fft, fft = -1 -> ifft
    assert(plan.fft == 1 || plan.fft == -1);

    const int num_warp = 8;
    const int n_cont[3] = {32, 16, 8};

    int step = 1;
    int RADIX = 1;
    dim3 threads, blocks;

    RADIX = plan.Ny;
    threads = {32, num_warp}; // 一个warp是32个线程，所以这里每个block里每一行都是一个warp，一共num_warp个warp（行）
    cudaFuncSetAttribute(plan.layer_0[plan.mergings[0]], cudaFuncAttributeMaxDynamicSharedMemorySize, RADIX * sizeof(half2) * n_cont[plan.mergings[0]]);
    LL nx_ny_nBatch = (LL)(plan.Nx) * (LL)(plan.Ny) * (LL)(plan.N_batch);
    LL LL_griddim = nx_ny_nBatch / (LL)(n_cont[plan.mergings[0]]);
    LL_griddim /= (LL)(RADIX);
    int _griddim = (int)LL_griddim;
    plan.layer_0[plan.mergings[0]]<<<_griddim, threads, RADIX * sizeof(half2) * n_cont[plan.mergings[0]], stream>>>((half2 *)data, plan.F_real, plan.F_imag, plan.fft);
    step *= RADIX;
    RADIX = plan.Nx;

    LL LL_blocks = nx_ny_nBatch / (LL)(step);
    LL_blocks /= (LL)(RADIX);
    int int_blocks = (int)LL_blocks;
    blocks = {step / n_cont[plan.mergings[1]], int_blocks};
    cudaFuncSetAttribute(plan.layer_1[plan.mergings[1]], cudaFuncAttributeMaxDynamicSharedMemorySize, RADIX * sizeof(half2) * n_cont[plan.mergings[1]]);
    plan.layer_1[plan.mergings[1]]<<<blocks, threads, RADIX * sizeof(half2) * n_cont[plan.mergings[1]], stream>>>(step, (half2 *)data, plan.F_real, plan.F_imag, plan.fft);
    step *= RADIX;
}
void fft_conv2d_half::TensorCoreFFTExec(fftPlan plan, half *data)
{
    // fft = 1 -> fft, fft = -1 -> ifft
    assert(plan.fft == 1 || plan.fft == -1);

    const int num_warp = 8;
    const int n_cont[3] = {32, 16, 8};

    int step = 1;
    int RADIX = 1;
    dim3 threads, blocks;

    RADIX = plan.Ny;
    threads = {32, num_warp}; // 一个warp是32个线程，所以这里每个block里每一行都是一个warp，一共num_warp个warp（行）
    cudaFuncSetAttribute(plan.layer_0[plan.mergings[0]], cudaFuncAttributeMaxDynamicSharedMemorySize, RADIX * sizeof(half2) * n_cont[plan.mergings[0]]);
    LL nx_ny_nBatch = (LL)(plan.Nx) * (LL)(plan.Ny) * (LL)(plan.N_batch);
    LL LL_griddim = nx_ny_nBatch / (LL)(n_cont[plan.mergings[0]]);
    LL_griddim /= (LL)(RADIX);
    int _griddim = (int)LL_griddim;
    plan.layer_0[plan.mergings[0]]<<<_griddim, threads, RADIX * sizeof(half2) * n_cont[plan.mergings[0]]>>>((half2 *)data, plan.F_real, plan.F_imag, plan.fft);
    step *= RADIX;
    RADIX = plan.Nx;

    LL LL_blocks = nx_ny_nBatch / (LL)(step);
    LL_blocks /= (LL)(RADIX);
    int int_blocks = (int)LL_blocks;
    blocks = {step / n_cont[plan.mergings[1]], int_blocks};
    cudaFuncSetAttribute(plan.layer_1[plan.mergings[1]], cudaFuncAttributeMaxDynamicSharedMemorySize, RADIX * sizeof(half2) * n_cont[plan.mergings[1]]);
    plan.layer_1[plan.mergings[1]]<<<blocks, threads, RADIX * sizeof(half2) * n_cont[plan.mergings[1]]>>>(step, (half2 *)data, plan.F_real, plan.F_imag, plan.fft);
    step *= RADIX;
}
void fft_conv2d_half::TensorCoreFFTCreate(fftPlan *plan, int nx, int ny, int n_batch, int fft, __GPU_ARCH__ GPU)
{
    plan->GPU = GPU;
    plan->Nx = nx;
    plan->Ny = ny;
    plan->N_batch = n_batch;
    plan->fft = fft;
    // setup functions
    const int num_warp = 8;
    const int n_cont_256 = 32;
    const int n_cont_512 = 16;
    const int n_cont_1024 = 8;
    if (GPU == __AMPERE__)
    {
        plan->layer_0[0] = layer_256_0_A100<n_cont_256, num_warp>;
        plan->layer_0[1] = layer_512_0_A100<n_cont_512, num_warp>;
        plan->layer_0[2] = layer_1024_0_A100<n_cont_1024, num_warp>;
        plan->layer_1[0] = layer_256_1_A100<n_cont_256, num_warp>;
        plan->layer_1[1] = layer_512_1_A100<n_cont_512, num_warp>;
    }
    else if (GPU == __VOLTA__)
    {
        plan->layer_0[0] = layer_256_0<n_cont_256, num_warp>;
        plan->layer_0[1] = layer_512_0<n_cont_512, num_warp>;
        plan->layer_0[2] = layer_1024_0<n_cont_1024, num_warp>;
        plan->layer_1[0] = layer_256_1<n_cont_256, num_warp>;
        plan->layer_1[1] = layer_512_1<n_cont_512, num_warp>;
    }
    // radices
    switch (nx)
    {
        case 256: plan->n_radices_x = 2; break;

        case 512:
            plan->n_radices_x = 3;
            plan->mergings[1] = 1;
            break;

        case 1024:
            plan->n_radices_x = 3;
            plan->radices_x[2] = 4;
            plan->mergings[1] = 2;
            break;

        default: break;
    }
    switch (ny)
    {
        case 256: plan->n_radices_y = 2; break;

        case 512:
            plan->n_radices_y = 3;
            plan->mergings[0] = 1;
            break;

        case 1024:
            plan->n_radices_y = 3;
            plan->radices_y[2] = 4;
            plan->mergings[0] = 2;
            break;

        default: break;
    }
    // F
    plan->F_real_tmp = (half *)malloc(sizeof(half) * 256);
    plan->F_imag_tmp = (half *)malloc(sizeof(half) * 256);

    if (plan->fft == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j)
            {
                plan->F_real_tmp[16 * i + j] = cosf(2 * M_PI * i * j / 16);
                plan->F_imag_tmp[16 * i + j] = -sinf(2 * M_PI * i * j / 16);
            }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j)
            {
                plan->F_real_tmp[16 * i + j] = cosf(2 * M_PI * i * j / 16);
                plan->F_imag_tmp[16 * i + j] = sinf(2 * M_PI * i * j / 16);
            }
    }
    cudaMalloc(&plan->F_real, sizeof(half) * 256);
    cudaMemcpy(plan->F_real, plan->F_real_tmp, sizeof(half) * 256, cudaMemcpyHostToDevice);
    cudaMalloc(&plan->F_imag, sizeof(half) * 256);
    cudaMemcpy(plan->F_imag, plan->F_imag_tmp, sizeof(half) * 256, cudaMemcpyHostToDevice);
}