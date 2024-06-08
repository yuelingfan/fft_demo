#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cstdlib>
#include <cuda_fp16.h>

half *data_host, *data_device;
half *data_host_2d, *data_device_2d;
cufftHandle plan_cu;
cufftHandle plan_cu_2d;

void setup_cu(double *data, int n, int n_batch)
{
    data_host = (half *)malloc(sizeof(half) * n * n_batch * 2);
    for (int i = 0; i < n_batch; ++i)
        for (int j = 0; j < n; ++j)
        {
            data_host[(j + i * n) * 2 + 0] = __float2half((float)data[0 + j * 2 + i * n * 2]);
            data_host[(j + i * n) * 2 + 1] = __float2half((float)data[1 + j * 2 + i * n * 2]);
        }

    cudaMalloc(&data_device, sizeof(half) * n * n_batch * 2);
    cudaMemcpy(data_device, data_host, sizeof(half) * n * n_batch * 2, cudaMemcpyHostToDevice);
    
    long long p_n[1];
    size_t worksize[1];
    p_n[0] = n;
    cufftCreate(&plan_cu);
    cufftXtMakePlanMany(plan_cu, 1, p_n, NULL, 0, 0, CUDA_C_16F, NULL, 0, 0, CUDA_C_16F, n_batch, worksize, CUDA_C_16F);
}

void doit_cu(int iter)
{
    for (int i = 0; i < iter; ++i)
        cufftXtExec(plan_cu, data_device, data_device, CUFFT_FORWARD);
    cudaDeviceSynchronize(); 
}

void finalize_cu(double *result,int n,int n_batch)
{
    cudaMemcpy(data_host, data_device, sizeof(half) * n * n_batch * 2, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_batch; ++i)
        for (int j = 0; j < n; ++j)
        {
            result[0 + j * 2 + i * n * 2] = __half2float(data_host[(j + i * n) * 2 + 0]);
            result[1 + j * 2 + i * n * 2] = __half2float(data_host[(j + i * n) * 2 + 1]);
        }
}

void setup_cu_2d(double *data, int nx, int ny, int n_batch)
{
    data_host_2d = (half *)malloc(sizeof(half) * nx * ny * n_batch * 2);
    for (int i = 0; i < n_batch; ++i)
        for (int j = 0; j < nx * ny; ++j)
        {
            data_host_2d[(j + i * nx * ny) * 2 + 0] = __float2half((float)data[0 + j * 2 + i * nx * ny * 2]);
            data_host_2d[(j + i * nx * ny) * 2 + 1] = __float2half((float)data[1 + j * 2 + i * nx * ny * 2]);
        }

    cudaMalloc(&data_device_2d, sizeof(half) * nx * ny * n_batch * 2);
    cudaMemcpy(data_device_2d, data_host_2d, sizeof(half) * nx * ny * n_batch * 2, cudaMemcpyHostToDevice);
    
    long long p_n[2];
    size_t worksize[1];
    p_n[0] = nx;
    p_n[1] = ny;
    cufftCreate(&plan_cu_2d);
    cufftXtMakePlanMany(plan_cu_2d, 2, p_n, NULL, 0, 0, CUDA_C_16F, NULL, 0, 0, CUDA_C_16F, n_batch, worksize, CUDA_C_16F);
}

void doit_cu_2d(int iter)
{
    for (int i = 0; i < iter; ++i)
        cufftXtExec(plan_cu_2d, data_device_2d, data_device_2d, CUFFT_FORWARD);
    cudaDeviceSynchronize(); 
}

void finalize_cu_2d(double *result,int nx,int ny,int n_batch)
{
    cudaMemcpy(data_host_2d, data_device_2d, sizeof(half) * nx * ny * n_batch * 2, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_batch; ++i)
        for (int j = 0; j < nx * ny; ++j)
        {
            result[0 + j * 2 + i * nx * ny * 2] = __half2float(data_host_2d[(j + i * nx * ny) * 2 + 0]);
            result[1 + j * 2 + i * nx * ny * 2] = __half2float(data_host_2d[(j + i * nx * ny) * 2 + 1]);
        }
}