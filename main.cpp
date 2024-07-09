#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <cstdio>
#include <fftconvolve2d.cuh>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cufft.h>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cudnn.h>
#include <random>
#include "tcfft_half.h"
#include "tcfft_half_2d.h"
#include <fftw3.h>
#include <iostream>
#include "tensorcore_fft_conv2d.h"
#include "utils.h"
#include "cufft.h"
#include <chrono>

using namespace std;

extern char *optarg;
extern int optopt;
#define NCHW 1

void generate_data(double *signal, double *filter, int s, int f, int C, int out, int seed = 42)
{
    srand(seed);
#pragma omp parallel for
    for (int i = 0; i < C; i++)
        for (int j = 0; j < s*s; j++)
        {
            signal[0 + j * 2 + i * s * 2] = 0.1f * (rand() % 10);
            signal[1 + j * 2 + i * s * 2] = 0.1f * (rand() % 10);
        }
#pragma omp parallel for
    for (int k = 0; k < out; k++)
    {
        for (int i = 0; i < C; i++)
        {
            for (int j = 0; j < f*f; j++)
            {
                filter[j+ i * f*f+ k*f*f*C] = 0.1f * (rand() % 10);
            }
        }
    }
}

void doubleToFloat(double* src, float* dest, int size) {
    for (int i = 0; i < size; ++i) {
        dest[i] = static_cast<float>(src[i]);
    }
}

double gettime()
{
    timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_usec * 1.0e-3 + tv.tv_sec * 1.0e3;
}

void fftw3_get_result(double *data, double *result, int n, int n_batch)
{
    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n);
    fftw_plan p = fftw_plan_dft_1d(n, in, in, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int i = 0; i < n_batch; ++i)
    {
        memcpy(in, data + 2 * i * n, sizeof(fftw_complex) * n);
        fftw_execute(p);
        memcpy(result + 2 * i * n, in, sizeof(fftw_complex) * n);
    }
    fftw_destroy_plan(p);
    fftw_free(in);
}

void fftw3_get_result_2d(double *data, double *result, int nx, int ny, int n_batch)
{
    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * nx * ny);
    fftw_plan p = fftw_plan_dft_2d(nx, ny, in, in, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int i = 0; i < n_batch; ++i)
    {
        memcpy(in, data + 2 * i * nx * ny, sizeof(fftw_complex) * nx * ny);
        fftw_execute(p);
        memcpy(result + 2 * i * nx * ny, in, sizeof(fftw_complex) * nx * ny);
    }
    fftw_destroy_plan(p);
    fftw_free(in);
}


void naive_fft_conv2d_ss_sf_valid(double *signal, double *filter, double *output, int sy, int sx, int fy, int fx)
{
    int output_size_x = sx - fx + 1;
    int output_size_y = sy - fy + 1;

    for (int i = 0; i < output_size_y; i++)
    {
        for (int j = 0; j < output_size_x; j++)
        {
            output[i * output_size_x + j] = 0.0f;
            for (int k = 0; k < fy; k++)
            {
                for (int l = 0; l < fx; l++)
                {
                    output[i * output_size_x + j] += filter[k * fx + l] * signal[(i + k) * sx + j + l];
                }
            }
        }
    }
}

double get_error(double *tested, double *standard, int n, int n_batch)
{
    double error = 0;
    for (int i = 0; i < n_batch; ++i)
#pragma omp parallel for reduction(+ \
                                   : error)
        for (int j = 0; j < n; ++j)
        {
            double tested_e = tested[0 + j * 2 + i * n * 2];
            double standard_e = standard[0 + j * 2 + i * n * 2];
            error += std::min(1.0, std::abs((tested_e - standard_e) / standard_e));
            tested_e = tested[1 + j * 2 + i * n * 2];
            standard_e = standard[1 + j * 2 + i * n * 2];
            error += std::min(1.0, std::abs((tested_e - standard_e) / standard_e));
        }
    return error / n / n_batch;
}

double get_error_conv(double *tested, double *standard, int nx, int ny, int n_batch, int scale = 1)
{
    double error = 0;
    for (int i = 0; i < n_batch; ++i)
#pragma omp parallel for reduction(+ : error)
        for (int j = 0; j < nx * ny; ++j)
        {
            double tested_e = tested[j + i * nx * ny];
            double standard_e = standard[j + i * nx * ny] * scale;
            error += std::min(1.0, std::abs((tested_e - standard_e) / (standard_e)));
        }
    return error / nx / ny / n_batch;
}

double get_error_f(float *tested, float *standard, int nx, int ny, int n_batch, int scale = 1)
{
    double error = 0;
    for (int i = 0; i < n_batch; ++i)
#pragma omp parallel for reduction(+ : error)
        for (int j = 0; j < nx * ny; ++j)
        {
            double tested_e = tested[j + i * nx * ny];
            double standard_e = standard[j + i * nx * ny] * scale;
            error += std::min(1.0, std::abs((tested_e - standard_e) / (standard_e)));
        }
    return error / nx / ny / n_batch;
}

void accuracy_1d(double *data, int n, int n_batch)
{
    double *standard = (double *)malloc(sizeof(double) * n * n_batch*2);
    double *tested_1d = (double *)malloc(sizeof(double) * n * n_batch * 2);
    double *tested_cu = (double *)malloc(sizeof(double) * n * n_batch * 2);
    fftw3_get_result(data, standard, n, n_batch);

    setup(data, n, n_batch);
    doit(1);
    finalize(tested_1d,n_batch);
    printf("tcfft1d error:%e\n", get_error(tested_1d, standard, n, n_batch));
    
    setup_cu(data, n, n_batch);
    doit_cu(1);
    finalize_cu(tested_cu,n,n_batch);
    printf("cufft1d error:%e\n", get_error(tested_cu, standard, n, n_batch));
}

void accuracy_2d(double *data, int nx, int ny, int n_batch)
{
    double *standard_2d = (double *)malloc(sizeof(double) * nx * ny * n_batch*2);
    double *tested_2d = (double *)malloc(sizeof(double) * nx * ny * n_batch * 2);
    double *tested_cu_2d = (double *)malloc(sizeof(double) * nx*ny * n_batch * 2);
    fftw3_get_result_2d(data, standard_2d, nx,ny, n_batch);

    setup_2d(data, nx, ny, n_batch);
    doit_2d(1);
    finalize_2d(tested_2d,n_batch);
    printf("tcfft2d error:%e\n", get_error(tested_2d, standard_2d, nx*ny, n_batch));

    setup_cu_2d(data, nx,ny, n_batch);
    doit_cu_2d(1);
    finalize_cu_2d(tested_cu_2d,nx,ny,n_batch);
    printf("cufft1d error:%e\n", get_error(tested_cu_2d, standard_2d, nx*ny, n_batch));
}

void speed(double *data, int nx,int ny,int n_batch)
{
    double *result = (double *)malloc(sizeof(double) * nx*ny * n_batch * 2);
    double *result_2d = (double *)malloc(sizeof(double) * nx*ny * n_batch * 2);
    double *result_cu = (double *)malloc(sizeof(double) * nx*ny * n_batch * 2);
    double *result_cu_2d = (double *)malloc(sizeof(double) * nx*ny * n_batch * 2);

    int iter,max_times = 1 << 30;;
    double t_min = 4;
    double run_time;
    setup(data, nx*ny, n_batch);
    for (int i = 1; i <= max_times; i <<= 1)
    {
        double t1;
        t1 = gettime();
        doit(i);
        run_time = gettime() - t1;
        iter = i;
        if (run_time > t_min)
            break;
    }
    finalize(result,n_batch);
    printf("tcfft_1d:iter: %d, time per iter: %lf\n",iter, run_time / iter);

    setup_2d(data, nx, ny, n_batch);
    for (int i = 1; i <= max_times; i <<= 1)
    {
        double t1;
        t1 = gettime();
        doit_2d(i);
        run_time = gettime() - t1;
        iter = i;
        if (run_time > t_min)
            break;
    }
    finalize_2d(result_2d,n_batch);
    printf("tcfft_2d:iter: %d, time per iter: %lf\n", iter, run_time / iter);

    setup_cu(data, nx*ny, n_batch);
    for (int i = 1; i <= max_times; i <<= 1)
    {
        double t1;
        t1 = gettime();
        doit_cu(i);
        run_time = gettime() - t1;
        iter = i;
        if (run_time > t_min)
            break;
    }
    finalize_cu(result_cu,nx*ny,n_batch);
    printf("cufft_1d:iter: %d, time per iter: %lf\n",iter, run_time / iter);

    setup_cu_2d(data, nx, ny, n_batch);
    for (int i = 1; i <= max_times; i <<= 1)
    {
        double t1;
        t1 = gettime();
        doit_cu_2d(i);
        run_time = gettime() - t1;
        iter = i;
        if (run_time > t_min)
            break;
    }
    finalize_cu_2d(result_cu_2d,nx,ny,n_batch);
    printf("cufft_2d:iter: %d, time per iter: %lf\n", iter, run_time / iter);
}

void speed_conv(double *signal, double *filter,int s, int f, int c, int out)
{
    INITTIMER
    double run_time;
    int output_tile_size=s - f + 1;
    fft_conv2d_half *conv2d_lyer;
    double *output_speed = (double *)malloc(sizeof(double) * output_tile_size * output_tile_size * out);

    for (int iter = 0; iter < 5; iter++)
    {
        conv2d_lyer = new fft_conv2d_half(signal, filter, c, out, s, s, f, f, __AMPERE__);
        START
        conv2d_lyer->conv2d_forward(output_speed);
        END_wo_print;
        std::cout << "our_fft_conv: time:" << milliseconds << std::endl;
        delete conv2d_lyer;
    }
}

void accuracy_conv(double *signal, double *filter, int s, int f, int C, int kernelCount,int fftSize)
{
    int output_tile_size=s - f + 1;
    double *output_standard = (double *)malloc(sizeof(double) *output_tile_size * output_tile_size * kernelCount);
    double *output_conv = (double *)malloc(sizeof(double) * output_tile_size * output_tile_size * kernelCount);
    float *direct_gpu_results = (float*)malloc(sizeof(float) * fftSize * kernelCount);
    float *turbo_v2_gpu_results = (float*)malloc(sizeof(float) * fftSize * kernelCount);
    float *signal_f = (float *)malloc(sizeof(float) * s * s * C);
    float *filter_f = (float *)malloc(sizeof(float) * f * f * kernelCount * C);

    doubleToFloat(signal,signal_f,s * s * C);
    doubleToFloat(filter,filter_f,f * f * kernelCount * C);

    /*Perform standard direct convolution on CPU*/
    naive_fft_conv2d_ss_sf_valid(signal, filter, output_standard, s, s, f, f);

    /*Perform our fft convolution on GPU*/
    fft_conv2d_half conv2d_lyer(signal, filter, C, kernelCount, s, s, f, f, __AMPERE__);
    conv2d_lyer.conv2d_forward(output_conv);

    /*Get accuracy*/
    double acc_conv = get_error_conv(output_conv, output_standard, output_tile_size, output_tile_size, 1);

    memset(direct_gpu_results, 0, sizeof(float) * fftSize * kernelCount);
    memset(turbo_v2_gpu_results, 0, sizeof(float) * fftSize * kernelCount);

    convolve_direct_fft(signal_f, signal_f, kernelCount, output_tile_size, output_tile_size, f, f, direct_gpu_results);
    convolve_turbo_fft_v2(signal_f, signal_f, kernelCount, output_tile_size, output_tile_size, f, f, turbo_v2_gpu_results);
    double acc_turbo = get_error_f(turbo_v2_gpu_results, direct_gpu_results, fftSize, kernelCount, 1);
    std::cout << "our_fft_half_conv error: " << acc_conv << std::endl;
    std::cout << "convolve_turbo_fft_v2 error: " << acc_turbo << std::endl;
}


int main(int argc, char *argv[])
{
    int nx = 256, ny = 256, n_batch = 1,C=1;
    int n=nx*ny;
    int kernelH = 13;
    int kernelW = 13;
    int dataH = nx - kernelH + 1;
    int dataW = ny - kernelW + 1;
    int kernelCount = 16;
    int fftH = dataH + kernelH - 1;
    int fftW = dataW + kernelW - 1;
    int fftSize = fftH * fftW;
    std::cout << "nx=ny= "<< nx << std::endl;
    std::cout << "n_batch: "<< n_batch << std::endl;
    std::cout << "kernelH: "<< kernelH << ",kernelH: "<< kernelH << std::endl;

    /*Generate input data*/
    double *data = (double *)malloc(sizeof(double) * nx * ny * C*2);
    double *signal = (double *)malloc(sizeof(double) * nx * ny * C);
    double *filter = (double *)malloc(sizeof(double) * kernelH * kernelH * kernelCount * C);
    

    generate_data(data, filter, nx, kernelH, C, kernelCount);
    memcpy(signal, data, nx * ny * C * sizeof(double));

    speed(data,nx,ny,n_batch);//tcfft
    speed_conv(signal, filter,nx,kernelH,C,kernelCount);//tcfftconv

    accuracy_conv(signal, filter,nx,kernelH,C,kernelCount,fftSize);//conv*3
    accuracy_2d(data,nx,ny,n_batch);//tcfft2d
    accuracy_1d(data,n,n_batch);//tcfft
    
    return 0;
}