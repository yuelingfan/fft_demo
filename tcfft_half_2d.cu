#include "tcfft_half_2d.h"
using namespace nvcuda;
const int WARP_SIZE = 32, WMMA_M = 16, WMMA_N = 16, WMMA_K = 16, CONT_SIZE_2D = 32;

__device__ inline void complex_mul(wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> &frag_F_real, wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> &frag_F_imag,
                                   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> &frag_in_real, wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> &frag_in_imag,
                                   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> &frag_out_real, wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> &frag_out_imag)
{
    wmma::fill_fragment(frag_out_real, 0.0);
    wmma::fill_fragment(frag_out_imag, 0.0);

    wmma::mma_sync(frag_out_real, frag_F_imag, frag_in_imag, frag_out_real);
    for (int i = 0; i < frag_out_real.num_elements; i++)
        frag_out_real.x[i] = -frag_out_real.x[i];
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
    for (int i = 0; i < frag_buf_real.num_elements; i++)
        frag_buf_real.x[i] = -frag_buf_real.x[i];
    wmma::mma_sync(frag_buf_real, frag_F_real, frag_in_real, frag_buf_real);
    for (int i = 0; i < frag_buf_real.num_elements; i++)
        frag_out_real.x[i] += frag_buf_real.x[i];

    wmma::mma_sync(frag_out_imag, frag_F_real, frag_in_imag, frag_out_imag);
    wmma::mma_sync(frag_out_imag, frag_F_imag, frag_in_real, frag_out_imag);
}

__device__ __host__ inline half2 W_N_K(int N, int K)
{
    half2 t = {cosf(2 * M_PI * K / N), -sinf(2 * M_PI * K / N)};
    return t;
}

__device__ __host__ inline float2 W_N_K_fp32(int N, int K)
{
    float2 t = {cosf(2 * M_PI * K / N), -sinf(2 * M_PI * K / N)};
    return t;
}

__device__ inline half2 const cmul(const half2 &a, const half2 &b)
{
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

__device__ inline half2 const cmul_mixed(const half2 &a, const float2 &b)
{
    return {a.x * __float2half(b.x) - a.y * __float2half(b.y), a.x * __float2half(b.y) + a.y * __float2half(b.x)};
}

__device__ inline void swap(half &a, half &b)
{
    half tmp = a;
    a = b;
    b = tmp;
}

template <int CONT_SIZE_2D, int NUM_WARP>
__global__ void layer_256_0(half2 *in, half *F_real, half *F_imag)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 256 * CONT_SIZE_2D;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_col = threadIdx.x / 16 * 4 + threadIdx.x % 16 / 8 * 8 + threadIdx.x % 4;
    half2 twiddle_unit = W_N_K(256, raw_col);

    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 16 * 16)
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
    /* opt test
    }
    __syncthreads();
    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        smem_in[eid / 32 * 32 + eid % 32 / 2 + eid % 32 % 2 * 16] = smem_in[eid];
    }
    __syncthreads();

    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 16 * 16)
    {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;
        int warp_start = i + threadIdx.y * 256;
    */

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
    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        in[block_start + eid] = smem_in[eid];
    }
}

template <int CONT_SIZE_2D, int NUM_WARP>
__global__ void layer_256_0_A100(half2 *in, half *F_real, half *F_imag)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 256 * CONT_SIZE_2D;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_row = threadIdx.x % 4 * 2;
    int raw_col = threadIdx.x / 4;
    // half2 twiddle_unit = W_N_K(256, raw_col);

    /* opt test
    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        smem_in[eid] = in[block_start + eid];
    }
    __syncthreads();

    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        smem_in[eid / 32 * 32 + eid % 32 / 2 + eid % 32 % 2 * 16] = smem_in[eid];
    }
    __syncthreads();
    */

    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 16 * 16)
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
            // half2 ele = smem_in[warp_start + row + col * 16]; // opt test
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        wmma::store_matrix_sync((half *)(smem_in + warp_start), frag_out_real, 16, wmma::mem_row_major);
        wmma::store_matrix_sync((half *)(smem_in + warp_start) + 256, frag_out_imag, 16, wmma::mem_row_major);

        wmma::load_matrix_sync(frag_in_real, (half *)(smem_in + warp_start), 16);
        wmma::load_matrix_sync(frag_in_imag, (half *)(smem_in + warp_start) + 256, 16);

        // half2 twiddle_factor = {1.0, 0};
        // for (int j = 0; j < 16; ++j)
        // {
        //     int row = j;
        //     int col = raw_col;
        //     half2 in_ele = {frag_in_real.x[j], frag_in_imag.x[j]};
        //     in_ele = cmul(in_ele, twiddle_factor);
        //     frag_in_real.x[j] = in_ele.x;
        //     frag_in_imag.x[j] = in_ele.y;
        //     twiddle_factor = cmul(twiddle_factor, twiddle_unit);
        // }
        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            half2 in_ele = {frag_in_real.x[j], frag_in_imag.x[j]};
            in_ele = cmul(in_ele, W_N_K(256, row * col));
            frag_in_real.x[8 + j] = frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = in_ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        // int raw_row = threadIdx.x / 16 * 4 + threadIdx.x % 8 / 4 * 8 + threadIdx.x % 4;
        // raw_col = threadIdx.x % 16 / 8 * 8;
        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            in[block_start + warp_start + row * 16 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
            // smem_in[warp_start + row * 16 + col] = {frag_out_real.x[j], frag_out_imag.x[j]}; //opt test
        }
    }

    /* opt test
    __syncthreads();
    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        in[block_start + eid] = smem_in[eid];
    }
    */
}

template <int CONT_SIZE_2D, int NUM_WARP>
__global__ void layer_256_1(int step, half2 *in, half *F_real, half *F_imag)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.y * step * 256 + blockIdx.x * CONT_SIZE_2D;

    int b_c_col = threadIdx.x / 16 * 4 + threadIdx.x % 16 / 8 * 8 + threadIdx.x % 4;
    // int glb_col = blockIdx.x * CONT_SIZE_2D + threadIdx.y % 2 * 16 + b_c_col;
    // half2 twiddle_unit = W_N_K(step * 16, glb_col);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    /* opt test
    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        smem_in[eid] = in[block_start + eid / CONT_SIZE_2D * step + eid % CONT_SIZE_2D];
    }
    __syncthreads();

    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        smem_in[eid / 32 * 32 + eid % 32 / 2 + eid % 32 % 2 * 16] = smem_in[eid];
    }
    __syncthreads();
    */

    for (int i_start = 0; i_start < 256 * CONT_SIZE_2D; i_start += NUM_WARP * 256)
    {
        int warp_start = i_start + threadIdx.y / 2 * 512 + threadIdx.y % 2 * 16;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;
        // half2 twiddle_factor = {1.0, 0};
        for (int j = 0; j < 16; ++j)
        {
            int col = b_c_col;
            int row = j;
            int eid = warp_start + row * 32 + col;
            half2 ele = in[block_start + eid / CONT_SIZE_2D * step + eid % CONT_SIZE_2D];
            // ele = cmul(ele, twiddle_factor);
            frag_in_real.x[j] = ele.x;
            frag_in_imag.x[j] = ele.y;
            // twiddle_factor = cmul(twiddle_factor, twiddle_unit);
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
    /* opt test
    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        smem_in[eid / 32 * 32 + eid % 32 / 2 + eid % 32 % 2 * 16] = smem_in[eid];
    }
    __syncthreads();
    */

    for (int i_start = 0; i_start < CONT_SIZE_2D / NUM_WARP; i_start++)
    {
        int warp_start = i_start * NUM_WARP * 16 + threadIdx.y * 16;
        // int glb_col_2 = blockIdx.x * CONT_SIZE_2D + i_start * step * 4 + threadIdx.y / 2 * step + threadIdx.y % 2 * 16 + b_c_col;
        int glb_col_2 = i_start * 4 + threadIdx.y / 2;
        // half2 twiddle_unit_2 = W_N_K(step * 256, glb_col_2);
        half2 twiddle_unit_2 = W_N_K(256, glb_col_2);
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

    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        in[block_start + eid / CONT_SIZE_2D * step + eid % CONT_SIZE_2D] = smem_in[eid];
    }
}

template <int CONT_SIZE_2D, int NUM_WARP>
__global__ void layer_256_1_A100(int step, half2 *in, half *F_real, half *F_imag)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.y * step * 256 + blockIdx.x * CONT_SIZE_2D;

    // int b_c_col = threadIdx.x / 16 * 4 + threadIdx.x % 16 / 8 * 8 + threadIdx.x % 4;
    // int glb_col = blockIdx.x * CONT_SIZE_2D + threadIdx.y % 2 * 16 + b_c_col;
    // half2 twiddle_unit = W_N_K(step * 16, glb_col);
    int warp_col = blockIdx.x * CONT_SIZE_2D + threadIdx.y % 2 * 16;

    int raw_row = threadIdx.x % 4 * 2;
    int raw_col = threadIdx.x / 4;

    half2 twiddle_factor;
    half2 twiddle_unit;
    // for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 32)
    // {
    //     int eid = i + t_block;
    //     smem_in[eid] = in[block_start + eid / CONT_SIZE_2D * step + eid % CONT_SIZE_2D];
    // }

    for (int i = 0; i < 2; ++i)
    {
        int eid = i * 512 * 8 + threadIdx.y * 512 + threadIdx.x;
        for (int j = 0; j < 16; ++j)
        {
            smem_in[eid] = in[block_start + eid / CONT_SIZE_2D * step + eid % CONT_SIZE_2D];
            eid += 32;
        }
    }

    __syncthreads();

    /* opt test
    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        smem_in[eid / 32 * 32 + eid % 32 / 2 + eid % 32 % 2 * 16] = smem_in[eid];
    }
    __syncthreads();
    */

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    half2 twiddle[8];
    // for (int j = 0; j < 8; ++j)
    // {
    //     int row = raw_row + j % 4 / 2 * 8 + j % 2;
    //     int col = raw_col + j / 4 * 8;
    //     twiddle[j] = W_N_K(step * 16, (warp_col + col) * row);
    // }

    for (int i_start = 0; i_start < 256 * CONT_SIZE_2D; i_start += NUM_WARP * 256)
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
            // ele = cmul(ele, twiddle[j]);
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
        }

        // half2 twiddle_factor = {1.0, 0};
        // for (int j = 0; j < 16; ++j)
        // {
        //     int col = b_c_col;
        //     int row = j;
        //     int eid = warp_start + row * 32 + col;
        //     half2 ele = in[block_start + eid / CONT_SIZE_2D * step + eid % CONT_SIZE_2D];
        //     ele = cmul(ele, twiddle_factor);
        //     frag_in_real.x[j] = ele.x;
        //     frag_in_imag.x[j] = ele.y;
        //     twiddle_factor = cmul(twiddle_factor, twiddle_unit);
        // }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            smem_in[warp_start + row * 32 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();

    // warp_col = blockIdx.x * CONT_SIZE_2D + threadIdx.y / 2 * step + threadIdx.y % 2 * 16;
    // for (int j = 0; j < 8; ++j)
    // {
    //     int row = raw_row + j % 4 / 2 * 8 + j % 2;
    //     int col = raw_col + j / 4 * 8;
    //     twiddle[j] = W_N_K(step * 256, warp_col * row + col * row);
    // }
    // half2 twiddle_unit_2[4];
    // for (int j = 0; j < 4; ++j)
    // {
    //     int row = raw_row + j / 2 * 8 + j % 2;
    //     twiddle_unit_2[j] = W_N_K(step * 256, step * 4 * row);
    // }

    for (int i = 0; i < 2; ++i)
    {
        twiddle_unit = W_N_K(256, threadIdx.y + i * 8);
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

    for (int i_start = 0; i_start < CONT_SIZE_2D / NUM_WARP; i_start++)
    {
        int warp_start = i_start * NUM_WARP * 16 + threadIdx.y * 16;
        // int glb_col_2 = blockIdx.x * CONT_SIZE_2D + c + threadIdx.y / 2 * step + threadIdx.y % 2 * 16 + b_c_col;
        // half2 twiddle_unit_2 = W_N_K(step * 256, glb_col_2);
        // warp_col = blockIdx.x * CONT_SIZE_2D + i_start * step * 4 + threadIdx.y / 2 * step + threadIdx.y % 2 * 16;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;
        // half2 twiddle_factor = {1.0, 0};
        // for (int j = 0; j < 16; ++j)
        // {
        //     int col = b_c_col;
        //     int row = j;
        //     half2 ele = smem_in[warp_start + row * 512 + col];
        //     ele = cmul(ele, twiddle_factor);
        //     frag_in_real.x[j] = ele.x;
        //     frag_in_imag.x[j] = ele.y;
        //     twiddle_factor = cmul(twiddle_factor, twiddle_unit_2);
        // }

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            half2 ele = smem_in[warp_start + row * 512 + col];
            // ele = cmul(ele, twiddle[j]);
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
            // twiddle[j] = cmul(twiddle[j], twiddle_unit_2[j % 4]);
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

    /* opt test
    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        smem_in[eid / 32 * 32 + eid % 32 / 2 + eid % 32 % 2 * 16] = smem_in[eid];
    }
    __syncthreads();
    */

    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        in[block_start + eid / CONT_SIZE_2D * step + eid % CONT_SIZE_2D] = smem_in[eid];
    }
}

template <int CONT_SIZE_2D, int NUM_WARP>
__global__ void layer_512_0(half2 *in, half *F_real, half *F_imag)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 512 * CONT_SIZE_2D;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_col = threadIdx.x / 16 * 4 + threadIdx.x % 16 / 8 * 8 + threadIdx.x % 4;
    half2 twiddle_unit = W_N_K(256, raw_col);
    half2 twiddle_two = W_N_K(512, t_block);

    for (int i = 0; i < 512 * CONT_SIZE_2D; i += NUM_WARP * 16 * 16)
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

    __syncthreads();
    for (int i = 0; i < 512 * CONT_SIZE_2D; i += NUM_WARP * 32 * 2)
    {
        int eid = i + t_block;
        half2 ele_0 = smem_in[eid];
        half2 ele_1 = cmul(smem_in[eid + 256], twiddle_two);
        in[block_start + eid] = __hadd2(ele_0, ele_1);
        in[block_start + eid + 256] = __hsub2(ele_0, ele_1);
    }
}

template <int CONT_SIZE_2D, int NUM_WARP>
__global__ void layer_512_0_A100(half2 *in, half *F_real, half *F_imag)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 512 * CONT_SIZE_2D;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_row = threadIdx.x % 4 * 2;
    int raw_col = threadIdx.x / 4;
    half2 twiddle_two = W_N_K(512, t_block);

    for (int i = 0; i < 512 * CONT_SIZE_2D; i += NUM_WARP * 16 * 16)
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
            in_ele = cmul(in_ele, W_N_K(256, row * col));
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
    for (int i = 0; i < 512 * CONT_SIZE_2D; i += NUM_WARP * 32 * 2)
    {
        int eid = i + t_block;
        half2 ele_0 = smem_in[eid];
        half2 ele_1 = cmul(smem_in[eid + 256], twiddle_two);
        in[block_start + eid] = __hadd2(ele_0, ele_1);
        in[block_start + eid + 256] = __hsub2(ele_0, ele_1);
    }
}

template <int CONT_SIZE_2D, int NUM_WARP>
__global__ void layer_512_1(int step, half2 *in, half *F_real, half *F_imag)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.y * step * 512 + blockIdx.x * CONT_SIZE_2D;

    int b_c_col = threadIdx.x / 16 * 4 + threadIdx.x % 16 / 8 * 8 + threadIdx.x % 4;
    // int glb_col = blockIdx.x * CONT_SIZE_2D + b_c_col;
    // half2 twiddle_unit = W_N_K(step * 16, glb_col);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    for (int i_start = 0; i_start < 512 * CONT_SIZE_2D; i_start += NUM_WARP * 256)
    {
        int warp_start = i_start + threadIdx.y * 256;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;
        // half2 twiddle_factor = {1.0, 0};
        for (int j = 0; j < 16; ++j)
        {
            int col = b_c_col;
            int row = j;
            int eid = warp_start + row * CONT_SIZE_2D + col;
            half2 ele = in[block_start + eid / CONT_SIZE_2D * step + eid % CONT_SIZE_2D];
            // ele = cmul(ele, twiddle_factor);
            frag_in_real.x[j] = ele.x;
            frag_in_imag.x[j] = ele.y;
            // twiddle_factor = cmul(twiddle_factor, twiddle_unit);
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);

        int acc_row = threadIdx.x / 16 * 4 + threadIdx.x % 8 / 4 * 8 + threadIdx.x % 4;
        int acc_col = threadIdx.x % 16 / 8 * 8;
        for (int j = 0; j < 8; ++j)
        {
            int row = acc_row;
            int col = j + acc_col;
            smem_in[warp_start + row * CONT_SIZE_2D + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();

    for (int i_start = 0; i_start < 4; i_start++)
    {
        int warp_start = i_start % 2 * NUM_WARP * 16 + i_start / 2 * 256 * CONT_SIZE_2D + threadIdx.y * 16;
        // int glb_col_2 = blockIdx.x * CONT_SIZE_2D + i_start % 2 * step * 8 + threadIdx.y * step + b_c_col;
        int glb_col_2 = i_start % 2 * 8 + threadIdx.y;
        // half2 twiddle_unit_2 = W_N_K(step * 256, glb_col_2);
        // half2 twiddle_unit_2 = W_N_K(256, glb_col_2);
        float2 twiddle_unit_2 = W_N_K_fp32(256, glb_col_2); // precision improved
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_in_imag;
        // half2 twiddle_factor = {1.0, 0};
        float2 twiddle_factor = {1.0, 0}; // precision improved
        for (int j = 0; j < 16; ++j)
        {
            int col = b_c_col;
            int row = j;
            half2 ele = smem_in[warp_start + row * 256 + col];
            // ele = cmul(ele, twiddle_factor);
            ele = cmul_mixed(ele, twiddle_factor); // precision improved
            frag_in_real.x[j] = ele.x;
            frag_in_imag.x[j] = ele.y;
            // twiddle_factor = cmul(twiddle_factor, twiddle_unit_2);
            twiddle_factor = cuCmulf(twiddle_factor, twiddle_unit_2); // precision improved
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

    // half2 twiddle_unit_2 = W_N_K(step * 512, 256 / CONT_SIZE_2D * step);
    // half2 twiddle_unit_2 = W_N_K(512, 256 / CONT_SIZE_2D);
    float2 twiddle_unit_2 = W_N_K_fp32(512, 256 / CONT_SIZE_2D); // precision improved
    // half2 twiddle_factor = W_N_K(step * 512, t_block / CONT_SIZE_2D * step + t_block % CONT_SIZE_2D);
    // half2 twiddle_factor = W_N_K(512, t_block / CONT_SIZE_2D);
    float2 twiddle_factor = W_N_K_fp32(512, t_block / CONT_SIZE_2D); // precision improved
    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        half2 ele_0 = smem_in[eid];
        // half2 ele_1 = cmul(smem_in[eid + 256 * CONT_SIZE_2D], twiddle_factor);
        half2 ele_1 = cmul_mixed(smem_in[eid + 256 * CONT_SIZE_2D], twiddle_factor); // precision improved
        in[block_start + eid / CONT_SIZE_2D * step + eid % CONT_SIZE_2D] = __hadd2(ele_0, ele_1);
        eid += 256 * CONT_SIZE_2D;
        in[block_start + eid / CONT_SIZE_2D * step + eid % CONT_SIZE_2D] = __hsub2(ele_0, ele_1);
        // twiddle_factor = cmul(twiddle_factor, twiddle_unit_2);
        twiddle_factor = cuCmulf(twiddle_factor, twiddle_unit_2); // precision improved
    }
}

template <int CONT_SIZE_2D, int NUM_WARP>
__global__ void layer_512_1_A100(int step, half2 *in, half *F_real, half *F_imag)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.y * step * 512 + blockIdx.x * CONT_SIZE_2D;

    // int b_c_col = threadIdx.x / 16 * 4 + threadIdx.x % 16 / 8 * 8 + threadIdx.x % 4;
    // int glb_col = blockIdx.x * CONT_SIZE_2D + b_c_col;
    // half2 twiddle_unit = W_N_K(step * 16, glb_col);

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
            smem_in[eid] = in[block_start + eid / CONT_SIZE_2D * step + eid % CONT_SIZE_2D];
            eid += 16;
        }
    }

    __syncthreads();

    for (int i_start = 0; i_start < 512 * CONT_SIZE_2D; i_start += NUM_WARP * 256)
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
            // ele = cmul(ele, twiddle[j]);
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
        twiddle_unit = W_N_K(256, threadIdx.y * 2 + threadIdx.x / 16);
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
        int warp_start = i_start % 2 * NUM_WARP * 16 + i_start / 2 * 256 * CONT_SIZE_2D + threadIdx.y * 16;
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

    half2 twiddle_unit_2 = W_N_K(512, 256 / CONT_SIZE_2D);
    twiddle_factor = W_N_K(512, t_block / CONT_SIZE_2D);
    for (int i = 0; i < 256 * CONT_SIZE_2D; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        half2 ele_0 = smem_in[eid];
        half2 ele_1 = cmul(smem_in[eid + 256 * CONT_SIZE_2D], twiddle_factor);
        in[block_start + eid / CONT_SIZE_2D * step + eid % CONT_SIZE_2D] = __hadd2(ele_0, ele_1);
        eid += 256 * CONT_SIZE_2D;
        in[block_start + eid / CONT_SIZE_2D * step + eid % CONT_SIZE_2D] = __hsub2(ele_0, ele_1);
        twiddle_factor = cmul(twiddle_factor, twiddle_unit_2);
    }
}

template <int CONT_SIZE_2D, int NUM_WARP>
__global__ void layer_1024_0(half2 *in, half *F_real, half *F_imag)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 1024 * CONT_SIZE_2D;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_col = threadIdx.x / 16 * 4 + threadIdx.x % 16 / 8 * 8 + threadIdx.x % 4;
    half2 twiddle_unit = W_N_K(256, raw_col);

    for (int i = 0; i < 1024 * CONT_SIZE_2D; i += NUM_WARP * 16 * 16)
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

    half2 twiddle_1024_1 = W_N_K(1024, t_block);
    half2 twiddle_1024_2 = cmul(twiddle_1024_1, twiddle_1024_1);
    half2 twiddle_1024_3 = cmul(twiddle_1024_2, twiddle_1024_1);

    __syncthreads();
    for (int i = 0; i < 1024 * CONT_SIZE_2D; i += NUM_WARP * 32 * 4)
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

template <int CONT_SIZE_2D, int NUM_WARP>
__global__ void layer_1024_0_A100(half2 *in, half *F_real, half *F_imag)
{
    extern __shared__ half2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 1024 * CONT_SIZE_2D;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_row = threadIdx.x % 4 * 2;
    int raw_col = threadIdx.x / 4;

    for (int i = 0; i < 1024 * CONT_SIZE_2D; i += NUM_WARP * 16 * 16)
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
            in_ele = cmul(in_ele, W_N_K(256, row * col));
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

    half2 twiddle_1024_1 = W_N_K(1024, t_block);
    half2 twiddle_1024_2 = cmul(twiddle_1024_1, twiddle_1024_1);
    half2 twiddle_1024_3 = cmul(twiddle_1024_2, twiddle_1024_1);

    __syncthreads();
    for (int i = 0; i < 1024 * CONT_SIZE_2D; i += NUM_WARP * 32 * 4)
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

void tcfftExec_2d(tcfftHandle_2d plan, half *data)
{
    const int num_warp = 8;
    const int n_cont[3] = {32, 16, 8};

    int step = 1;
    int RADIX = 1;
    dim3 threads, blocks;

    // V100
    RADIX = plan.Ny;
    threads = {32, num_warp};
    cudaFuncSetAttribute(plan.layer_0[plan.mergings[0]], cudaFuncAttributeMaxDynamicSharedMemorySize, RADIX * sizeof(half2) * n_cont[plan.mergings[0]]);
    plan.layer_0[plan.mergings[0]]<<<plan.Nx * plan.Ny * plan.N_batch / n_cont[plan.mergings[0]] / RADIX, threads, RADIX * sizeof(half2) * n_cont[plan.mergings[0]]>>>((half2 *)data, plan.F_real, plan.F_imag);
    step *= RADIX;

    RADIX = plan.Nx;
    blocks = {step / n_cont[plan.mergings[1]], plan.N_batch * plan.Nx * plan.Ny / step / RADIX};
    cudaFuncSetAttribute(plan.layer_1[plan.mergings[1]], cudaFuncAttributeMaxDynamicSharedMemorySize, RADIX * sizeof(half2) * n_cont[plan.mergings[1]]);
    plan.layer_1[plan.mergings[1]]<<<blocks, threads, RADIX * sizeof(half2) * n_cont[plan.mergings[1]]>>>(step, (half2 *)data, plan.F_real, plan.F_imag);
    step *= RADIX;
}

void tcfftCreate_2d(tcfftHandle_2d *plan, int nx, int ny, int n_batch)
{
    plan->Nx = nx;
    plan->Ny = ny;
    plan->N_batch = n_batch;
    // setup functions
    const int num_warp = 8;
    const int n_cont_256 = 32;
    const int n_cont_512 = 16;
    const int n_cont_1024 = 8;
    plan->layer_0[0] = layer_256_0<n_cont_256, num_warp>;
    plan->layer_0[1] = layer_512_0<n_cont_512, num_warp>;
    plan->layer_0[2] = layer_1024_0<n_cont_1024, num_warp>;
    plan->layer_1[0] = layer_256_1<n_cont_256, num_warp>;
    plan->layer_1[1] = layer_512_1<n_cont_512, num_warp>;
    // radices
    switch (nx)
    {
    case 256:
        plan->n_radices_x = 2;
        break;

    case 512:
        plan->n_radices_x = 3;
        plan->mergings[1] = 1;
        break;

    case 1024:
        plan->n_radices_x = 3;
        plan->radices_x[2] = 4;
        plan->mergings[1] = 2;
        break;

    default:
        break;
    }
    switch (ny)
    {
    case 256:
        plan->n_radices_y = 2;
        break;

    case 512:
        plan->n_radices_y = 3;
        plan->mergings[0] = 1;
        break;

    case 1024:
        plan->n_radices_y = 3;
        plan->radices_y[2] = 4;
        plan->mergings[0] = 2;
        break;

    default:
        break;
    }
    // F
    plan->F_real_tmp = (half *)malloc(sizeof(half) * 256);
    plan->F_imag_tmp = (half *)malloc(sizeof(half) * 256);
#pragma omp parallel for
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
        {
            plan->F_real_tmp[16 * i + j] = cosf(2 * M_PI * i * j / 16);
            plan->F_imag_tmp[16 * i + j] = -sinf(2 * M_PI * i * j / 16);
        }
    cudaMalloc(&plan->F_real, sizeof(half) * 256);
    cudaMemcpy(plan->F_real, plan->F_real_tmp, sizeof(half) * 256, cudaMemcpyHostToDevice);
    cudaMalloc(&plan->F_imag, sizeof(half) * 256);
    cudaMemcpy(plan->F_imag, plan->F_imag_tmp, sizeof(half) * 256, cudaMemcpyHostToDevice);
}

int *rev_x, *rev_y,Nx, Ny;
half *in_host_2d, *in_device_1;
tcfftHandle_2d plan_2d;

void gen_rev_2d(int N, int rev[], int radices[], int n_radices)
{
    int *tmp_0 = (int *)malloc(sizeof(int) * N);
    int *tmp_1 = (int *)malloc(sizeof(int) * N);
    int now_N = N;
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
        tmp_0[i] = i;
    for (int i = n_radices - 1; i >= 0; --i)
    {
#pragma omp parallel for
        for (int j = 0; j < N; j += now_N)
            for (int k = 0; k < radices[i]; ++k)
                for (int l = 0; l < now_N / radices[i]; ++l)
                {
                    tmp_1[j + l + k * (now_N / radices[i])] = tmp_0[j + l * radices[i] + k];
                }
        now_N /= radices[i];
        std::swap(tmp_0, tmp_1);
    }
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
        rev[i] = tmp_0[i];
}


void setup_2d(double *data, int nx, int ny, int n_batch)
{
    Nx = nx;
    Ny = ny;
    tcfftCreate_2d(&plan_2d, nx, Ny, n_batch);
 // in_host
    rev_x = (int *)malloc(sizeof(int) * Nx);
    rev_y = (int *)malloc(sizeof(int) * Ny);
    gen_rev_2d(Nx, rev_x, plan_2d.radices_x, plan_2d.n_radices_x);
    gen_rev_2d(Ny, rev_y, plan_2d.radices_y, plan_2d.n_radices_y);
    in_host_2d = (half *)malloc(sizeof(half) * 2 * Nx * Ny * n_batch);
#pragma omp parallel for
    for (int i = 0; i < n_batch; ++i)
        for (int j = 0; j < Nx; ++j)
            for (int k = 0; k < Ny; ++k)
            {
                in_host_2d[2 * (i * Nx * Ny + j * Ny + k) + 0] = data[2 * (i * Nx * Ny + rev_x[j] * Ny + rev_y[k]) + 0];
                in_host_2d[2 * (i * Nx * Ny + j * Ny + k) + 1] = data[2 * (i * Nx * Ny + rev_x[j] * Ny + rev_y[k]) + 1];
            }
    cudaMalloc(&in_device_1, sizeof(half) * Nx * Ny * n_batch * 2);
    cudaMemcpy(in_device_1, in_host_2d, sizeof(half) * Nx * Ny * n_batch * 2, cudaMemcpyHostToDevice);
}

void finalize_2d(double *result,int n_batch)
{
    cudaMemcpy(in_host_2d, in_device_1, sizeof(half) * Nx * Ny * n_batch * 2, cudaMemcpyDeviceToHost);
#pragma omp paralllel for
    for (int i = 0; i < n_batch * Nx * Ny; ++i)
    {
        result[2 * i + 0] = in_host_2d[2 * i + 0];
        result[2 * i + 1] = in_host_2d[2 * i + 1];
    }
}

void doit_2d(int iter)
{
    for (int t = 0; t < iter; ++t)
        tcfftExec_2d(plan_2d, in_device_1);
    cudaDeviceSynchronize();
}