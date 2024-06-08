#include "tensorcore_fft_conv2d.h"
#include "utils.h"
#include <array>
#include <fstream>
#include <iostream>
#include <string.h>

#define ULL unsigned long long int

fft_conv2d_half::fft_conv2d_half() { cudaDeviceSynchronize(); }

fft_conv2d_half::fft_conv2d_half(double *signal, double *filter, int S, int F, int sy, int sx, int fy, int fx, __GPU_ARCH__ GPU)
{
    this->signal = signal;
    this->filter = filter;
    this->S = S;
    this->F = F;
    this->sx = sx;
    this->sy = sy;
    this->fx = fx;
    this->fy = fy;
    this->GPU = GPU;
    if (sx + fx - 1 <= 256)
    {
        this->padded_x = 256;
        this->padded_y = 256;
    }
    else
    {
        this->padded_x = 512;
        this->padded_y = 512;
    }
    this->output_x = sx - fx + 1;
    this->output_y = sy - fy + 1;
    cudaMalloc(&cu_signal, sizeof(double) * sx * sy * S);
    cudaMalloc(&cu_filter, sizeof(double) * fx * fy * F * S);
    cudaMalloc(&cu_signal_padded, sizeof(half) * padded_x * padded_y * 2 * S);
    cudaMalloc(&cu_filter_padded, sizeof(half) * padded_x * padded_y * 2 * F * S);
    cudaMemset(cu_signal_padded, 0, sizeof(half) * padded_x * padded_y * 2 * S);
    cudaMemset(cu_filter_padded, 0, sizeof(half) * padded_x * padded_y * 2 * F * S);
    cudaMemcpy(cu_signal, signal, sizeof(double) * sx * sy * S, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_filter, filter, sizeof(double) * fx * fy * F * S, cudaMemcpyHostToDevice);
    TensorCoreFFTCreate(&plan_signal, this->padded_x, this->padded_y, S, 1, GPU);
    TensorCoreFFTCreate(&plan_filter, this->padded_x, this->padded_y, F * S, 1, GPU);
    TensorCoreFFTCreate(&plan_pointwise, this->padded_x, this->padded_y, F, -1, GPU);

    cudaMalloc(&rev_x_cu, sizeof(int) * 1024);
    cudaMalloc(&rev_y_cu, sizeof(int) * 1024);

    gen_rev(padded_x, padded_y);

    fft_conv2d_inv_rev_pad();

    TensorCoreFFTExec(plan_signal, cu_signal_padded);

    TensorCoreFFTExec(plan_filter, cu_filter_padded);
    cudaDeviceSynchronize();
}
fft_conv2d_half::fft_conv2d_half(int in_channel, int out_channel, int kernel_size, __GPU_ARCH__ GPU)
{
    this->S = in_channel;
    this->F = out_channel;
    this->fx = kernel_size;
    this->fy = kernel_size;
    this->GPU = GPU;
    cudaDeviceSynchronize();
}

fft_conv2d_half::~fft_conv2d_half()
{
    cudaFree(cu_signal);
    cudaFree(cu_filter);
    cudaFree(cu_signal_padded);
    cudaFree(cu_filter_padded);
    cudaFree(rev_x_cu);
    cudaFree(rev_y_cu);
    cudaDeviceSynchronize();
}
void fft_conv2d_half::fft_2d_c2c(double *input, double *output, int x, int y, int batch, __GPU_ARCH__ GPU, int num_streams, bool benchmark)
{
    INITTIMER

    // stream init
    int batches_per_stream = batch / num_streams;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    fftPlan plan;
    TensorCoreFFTCreate(&plan, x, y, batches_per_stream, 1, GPU);

    double *cu_input_double;
    half *cu_input;

    cudaMalloc(&cu_input_double, sizeof(double) * x * y * batch * 2);
    cudaMalloc(&cu_input, sizeof(half) * x * y * batch * 2);

    cudaDeviceSynchronize();

    for (int i = 0; i < num_streams; ++i)
    {
        ULL offset = (ULL)i * (ULL)batches_per_stream * x * y * 2;

        cudaMemcpy3DParms params = {0};
        params.srcPtr = make_cudaPitchedPtr(input + offset, x * 2 * sizeof(double), x * 2, y);
        params.dstPtr = make_cudaPitchedPtr(cu_input_double + offset, x * 2 * sizeof(double), x * 2, y);
        params.extent = make_cudaExtent(x * 2 * sizeof(double), y, batches_per_stream);
        params.kind = cudaMemcpyHostToDevice;

        cudaMemcpy3DAsync(&params, streams[i]);
        rev(cu_input_double + offset, cu_input + offset, x, y, batches_per_stream, offset, streams[i]);

        TensorCoreFFTExec(plan, cu_input + offset, streams[i]);

        half2double(cu_input + offset, cu_input_double + offset, x, y, 2, batches_per_stream, streams[i]);
    }

    // 使用 cudaMemcpy3DAsync 将数据从设备复制回主机
    for (int i = 0; i < num_streams; ++i)
    {
        size_t offset = i * batches_per_stream * y * x * 2;

        cudaMemcpy3DParms params = {0};
        params.srcPtr = make_cudaPitchedPtr(cu_input_double + offset, x * 2 * sizeof(double), x * 2, y);
        params.dstPtr = make_cudaPitchedPtr(output + offset, x * 2 * sizeof(double), x * 2, y);
        params.extent = make_cudaExtent(x * 2 * sizeof(double), y, batches_per_stream);
        params.kind = cudaMemcpyDeviceToHost;

        cudaMemcpy3DAsync(&params, streams[i]);
    }
    // 等待所有流完成
    for (int i = 0; i < num_streams; ++i)
    {
        cudaStreamSynchronize(streams[i]);
    }

    // 释放内存和 CUDA 流
    cudaFree(cu_input);
    cudaFree(cu_input_double);
    for (int i = 0; i < num_streams; ++i)
    {
        cudaStreamDestroy(streams[i]);
    }
}
void fft_conv2d_half::ifft_2d_c2c(double *input, double *output, int x, int y, int batch, __GPU_ARCH__ GPU, int num_streams, bool benchmark)
{
    INITTIMER

    // stream init
    int batches_per_stream = batch / num_streams;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    fftPlan plan;
    TensorCoreFFTCreate(&plan, x, y, batches_per_stream, -1, GPU);

    double *cu_input_double;
    half *cu_input;

    cudaMalloc(&cu_input_double, sizeof(double) * x * y * batch * 2);
    cudaMalloc(&cu_input, sizeof(half) * x * y * batch * 2);

    cudaDeviceSynchronize();

    for (int i = 0; i < num_streams; ++i)
    {
        ULL offset = (ULL)i * (ULL)batches_per_stream * x * y * 2;

        cudaMemcpy3DParms params = {0};
        params.srcPtr = make_cudaPitchedPtr(input + offset, x * 2 * sizeof(double), x * 2, y);
        params.dstPtr = make_cudaPitchedPtr(cu_input_double + offset, x * 2 * sizeof(double), x * 2, y);
        params.extent = make_cudaExtent(x * 2 * sizeof(double), y, batches_per_stream);
        params.kind = cudaMemcpyHostToDevice;

        cudaMemcpy3DAsync(&params, streams[i]);
        rev(cu_input_double + offset, cu_input + offset, x, y, batches_per_stream, offset, streams[i]);

        TensorCoreFFTExec(plan, cu_input + offset, streams[i]);

        half2double(cu_input + offset, cu_input_double + offset, x, y, 2, batches_per_stream, streams[i]);
    }

    // 使用 cudaMemcpy3DAsync 将数据从设备复制回主机
    for (int i = 0; i < num_streams; ++i)
    {
        size_t offset = i * batches_per_stream * y * x * 2;

        cudaMemcpy3DParms params = {0};
        params.srcPtr = make_cudaPitchedPtr(cu_input_double + offset, x * 2 * sizeof(double), x * 2, y);
        params.dstPtr = make_cudaPitchedPtr(output + offset, x * 2 * sizeof(double), x * 2, y);
        params.extent = make_cudaExtent(x * 2 * sizeof(double), y, batches_per_stream);
        params.kind = cudaMemcpyDeviceToHost;

        cudaMemcpy3DAsync(&params, streams[i]);
    }
    // 等待所有流完成
    for (int i = 0; i < num_streams; ++i)
    {
        cudaStreamSynchronize(streams[i]);
    }

    // 释放内存和 CUDA 流
    cudaFree(cu_input);
    cudaFree(cu_input_double);
    for (int i = 0; i < num_streams; ++i)
    {
        cudaStreamDestroy(streams[i]);
    }
}
void fft_conv2d_half::tmp_test()
{
    int nx = 256, ny = 256;
    int idx1, idx2;
    fftPlan plan;
    double *data = (double *)malloc(sizeof(double) * nx * ny * 2);
    for (int i = 0; i < 1; ++i)
        // #pragma omp parallel for
        for (int j = 0; j < nx * ny; ++j)
        {
            idx1 = (int)j * 2 + (int)i * (int)nx * ny * 2;
            idx2 = idx1 + 1;
            data[idx1] = 0.0001f * (j % 10);
            data[idx2] = 0.0001f * (j % 10);
            //   data[0 + j * 2 + i * n * 2] = 0.0001f * rand() / RAND_MAX;
            //   data[1 + j * 2 + i * n * 2] = 0.0001f * rand() / RAND_MAX;
        }
    TensorCoreFFTCreate(&plan, 256, 256, 1, -1, GPU);
}
std::string fft_conv2d_half::toString() { return "test"; }

void fft_conv2d_half::gen_rev(int Nx, int Ny)
{
    if (Nx == 256)
    {
        cudaMemcpy(rev_x_cu, rev_256, sizeof(int) * 1024, cudaMemcpyHostToDevice);
        cudaMemcpy(rev_y_cu, rev_256, sizeof(int) * 1024, cudaMemcpyHostToDevice);
    }
    else if (Nx == 512)
    {
        cudaMemcpy(rev_x_cu, rev_512, sizeof(int) * 1024, cudaMemcpyHostToDevice);
        cudaMemcpy(rev_y_cu, rev_512, sizeof(int) * 1024, cudaMemcpyHostToDevice);
    }
}

void fft_conv2d_half::TENSORCORE_FFT_CONV_2D()
{
    int Nx = this->padded_x;
    int Ny = this->padded_y;

    cudaMalloc(&cu_pointwise_half, sizeof(half) * Nx * Ny * F * 2);
    cudaMalloc(&cu_pointwise_float, sizeof(float) * Nx * Ny * F * 2);
    cudaMalloc(&max_per_line, sizeof(float) * F * Ny);
    cudaMalloc(&max_per_tile, sizeof(float) * F);

    fft_pointwise();
    cudaFree(max_per_line);
    cudaFree(cu_pointwise_float);

    TensorCoreFFTExec(plan_pointwise, cu_pointwise_half);
    cudaDeviceSynchronize();

    cudaMalloc(&cu_conv2d_forward_result_padded, sizeof(double) * Nx * Ny * F * 2);

    fft_conv_final_stage();
    cudaDeviceSynchronize();
    cudaFree(cu_pointwise_half);
    cudaFree(max_per_tile);
}

void fft_conv2d_half::readout_fwd(bool freeCuMem)
{
    double *cu_output;
    cudaMalloc(&cu_output, sizeof(double) * output_x * output_y * F);
    read_out_scale_double2double(cu_conv2d_forward_result_padded, cu_output, output, padded_x, padded_y, output_x, output_y, F, (1.0f / (this->padded_x * this->padded_y)));
    cudaFree(cu_output);
    cudaFree(cu_conv2d_forward_result_padded);
}
void fft_conv2d_half::conv2d_forward(double *output)
{
    this->output = output;

    TENSORCORE_FFT_CONV_2D();

    readout_fwd();
}
void fft_conv2d_half::conv2d_forward(double *signal, double *filter, double *output, int S, int F, int sy, int sx, int fy, int fx, __GPU_ARCH__ GPU)
{
    this->signal = signal;
    this->filter = filter;
    this->output = output;
    this->S = S;
    this->F = F;
    this->sx = sx;
    this->sy = sy;
    this->fx = fx;
    this->fy = fy;
    this->GPU = GPU;
    if (sx + fx - 1 <= 256)
    {
        this->padded_x = 256;
        this->padded_y = 256;
    }
    else
    {
        this->padded_x = 512;
        this->padded_y = 512;
    }
    this->output_x = sx - fx + 1;
    this->output_y = sy - fy + 1;
    cudaMalloc(&cu_signal, sizeof(double) * sx * sy * S);
    cudaMalloc(&cu_filter, sizeof(double) * fx * fy * F * S);
    cudaMalloc(&cu_signal_padded, sizeof(half) * padded_x * padded_y * 2 * S);
    cudaMalloc(&cu_filter_padded, sizeof(half) * padded_x * padded_y * 2 * F * S);
    cudaMemset(cu_signal_padded, 0, sizeof(half) * padded_x * padded_y * 2 * S);
    cudaMemset(cu_filter_padded, 0, sizeof(half) * padded_x * padded_y * 2 * F * S);
    cudaMemcpy(cu_signal, signal, sizeof(double) * sx * sy * S, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_filter, filter, sizeof(double) * fx * fy * F * S, cudaMemcpyHostToDevice);
    TensorCoreFFTCreate(&plan_signal, this->padded_x, this->padded_y, S, 1, GPU);
    TensorCoreFFTCreate(&plan_filter, this->padded_x, this->padded_y, F * S, 1, GPU);
    TensorCoreFFTCreate(&plan_pointwise, this->padded_x, this->padded_y, F, -1, GPU);

    cudaMalloc(&rev_x_cu, sizeof(int) * 1024);
    cudaMalloc(&rev_y_cu, sizeof(int) * 1024);

    gen_rev(padded_x, padded_y);

    fft_conv2d_inv_rev_pad();

    TensorCoreFFTExec(plan_signal, cu_signal_padded);
    TensorCoreFFTExec(plan_filter, cu_filter_padded);
    cudaDeviceSynchronize();
    cudaFree(cu_signal);
    cudaFree(cu_filter);

    int Nx = this->padded_x;
    int Ny = this->padded_y;

    cudaMalloc(&cu_pointwise_half, sizeof(half) * Nx * Ny * F * 2);
    cudaMalloc(&cu_pointwise_float, sizeof(float) * Nx * Ny * F * 2);
    cudaMalloc(&max_per_line, sizeof(float) * F * Ny);
    cudaMalloc(&max_per_tile, sizeof(float) * F);

    fft_pointwise();
    cudaDeviceSynchronize();
    cudaFree(max_per_line);
    cudaFree(cu_pointwise_float);

    TensorCoreFFTExec(plan_pointwise, cu_pointwise_half);
    cudaDeviceSynchronize();

    cudaMalloc(&cu_conv2d_forward_result_padded, sizeof(double) * Nx * Ny * F * 2);

    fft_conv_final_stage();
    cudaDeviceSynchronize();
    cudaFree(cu_pointwise_half);
    cudaFree(max_per_tile);
    readout_fwd();
}
void fft_conv2d_half::conv2d_backward(double *delta_in)
{
    /*COMPUTE DELTA FOR FILTER*/

    int delta_in_x = this->output_x;
    int delta_in_y = this->output_y;
    int delta_padded_x = (2 * sx - fx) <= 256 ? 256 : 512;

    gen_rev(delta_padded_x, delta_padded_x);

    TensorCoreFFTCreate(&plan_signal_for_bp_filter, delta_padded_x, delta_padded_x, S, 1, GPU);
    TensorCoreFFTCreate(&plan_delta_in_for_bp_filter, delta_padded_x, delta_padded_x, F, 1, GPU);
    TensorCoreFFTCreate(&plan_pointwise_for_bp_filter, delta_padded_x, delta_padded_x, F * S, -1, GPU);

    cudaMalloc(&cu_delta_in, sizeof(double) * delta_in_x * delta_in_x * F);
    cudaMalloc(&cu_delta_in_padded, sizeof(half) * delta_padded_x * delta_padded_x * 2 * F);
    cudaMalloc(&cu_signal_padded, sizeof(half) * delta_padded_x * delta_padded_x * 2 * S);
    cudaMemcpy(cu_delta_in, delta_in, sizeof(double) * delta_in_x * delta_in_x * F, cudaMemcpyHostToDevice);
    cudaMalloc(&cu_pointwise_for_delta_filter, sizeof(half) * delta_padded_x * delta_padded_x * S * F * 2);

    inv_pad_deltain_signal(delta_padded_x);

    TensorCoreFFTExec(plan_signal_for_bp_filter, cu_signal_padded);
    TensorCoreFFTExec(plan_delta_in_for_bp_filter, cu_delta_in_padded);

    pointwise_for_delta_weight(delta_padded_x);
    cudaFree(cu_signal_padded);
    cudaFree(cu_delta_in_padded);

    TensorCoreFFTExec(plan_pointwise_for_bp_filter, cu_pointwise_for_delta_filter);

    double *cu_pointwise_for_delta_filter_double;
    cudaMalloc(&cu_pointwise_for_delta_filter_double, sizeof(double) * fx * fy * S * F);
    delta_for_filter = (double *)malloc(sizeof(double) * fy * fx * F * S);

    read_out_scale_half2double(cu_pointwise_for_delta_filter, cu_pointwise_for_delta_filter_double, delta_for_filter, delta_padded_x, delta_padded_x, fx, fy, S * F, (1.0f / (delta_padded_x * delta_padded_x)));
    cudaFree(cu_pointwise_for_delta_filter_double);
    cudaFree(cu_pointwise_for_delta_filter);

    /*COMPUTE DELTA FOR SIGNAL*/

    delta_in_x = sx + fx - 1;
    delta_padded_x = (sx + 2 * fx - 2) <= 256 ? 256 : 512;

    TensorCoreFFTCreate(&plan_filter_for_bp_signal, delta_padded_x, delta_padded_x, F * S, 1, GPU);
    TensorCoreFFTCreate(&plan_delta_in_for_bp_signal, delta_padded_x, delta_padded_x, F, 1, GPU);
    TensorCoreFFTCreate(&plan_pointwise_for_bp_signal, delta_padded_x, delta_padded_x, S, -1, GPU);

    double *cu_delta_in_0_padded;
    cudaMalloc(&cu_delta_in_0_padded, sizeof(double) * delta_in_x * delta_in_x * F);

    pad_fx_1_circles_of_0(cu_delta_in_0_padded, fx - 1, output_x, output_y);
    cudaFree(cu_delta_in);

    cudaMalloc(&cu_delta_in_padded, sizeof(half) * delta_padded_x * delta_padded_x * 2 * F);
    cudaMalloc(&cu_filter_padded, sizeof(half) * delta_padded_x * delta_padded_x * 2 * F * S);

    fft_conv2d_rev_pad(cu_delta_in_0_padded, cu_filter, cu_delta_in_padded, cu_filter_padded, delta_padded_x, delta_in_x, fx, F, F * S);
    cudaFree(cu_delta_in_0_padded);
    cudaFree(cu_filter);

    TensorCoreFFTExec(plan_filter_for_bp_signal, cu_filter_padded);
    TensorCoreFFTExec(plan_delta_in_for_bp_signal, cu_delta_in_padded);

    cudaMalloc(&cu_pointwise_for_delta_signal, sizeof(half) * delta_padded_x * delta_padded_x * 2 * S);
    cudaMemset(cu_pointwise_for_delta_signal, 0, sizeof(half) * delta_padded_x * delta_padded_x * 2 * S);

    pointwise_for_delta_input(delta_padded_x, cu_pointwise_for_delta_signal);
    cudaFree(cu_filter_padded);
    cudaFree(cu_delta_in_padded);

    TensorCoreFFTExec(plan_pointwise_for_bp_signal, cu_pointwise_for_delta_signal);

    delta_for_signal = (double *)malloc(sizeof(double) * sx * sy * S);
    double *cu_delta_for_signal;
    cudaMalloc(&cu_delta_for_signal, sizeof(double) * sx * sy * S);
    read_out_scale_half2double(cu_pointwise_for_delta_signal, cu_delta_for_signal, delta_for_signal, delta_padded_x, delta_padded_x, sx, sy, S, (1.0f / (delta_padded_x * delta_padded_x)));
    cudaFree(cu_delta_for_signal);
}

void fft_conv2d_half::printplan(fftPlan plan)
{
    printf("fft:%d\n\
    Nx, Ny, N_batch:%d%d%d\n\
    radices_x_1, radices_x_2, radices_x_3:%d%d%d\n\
    radices_y_1, radices_y_2, radices_y_3:%d%d%d\n\
    n_radices_x,n_radices_y:%d%d\n\
    mergings_0,mergings_1:%d%d\n\
    \n",
           plan.fft, plan.Nx, plan.Ny, plan.N_batch, plan.radices_x[0], plan.radices_x[1], plan.radices_x[2], plan.radices_y[0], plan.radices_y[1], plan.radices_y[2], plan.n_radices_x, plan.n_radices_y, plan.mergings[0],
           plan.mergings[1]);
}