#include "tensorcore_fft_conv2d.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cudnn.h>
#include <unistd.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <fftconvolve2d.cuh>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cufft.h>

extern char *optarg;
extern int optopt;
#define NCHW 1

using namespace std;

void generate_data(double *signal, double *filter, int s, int f, int C, int out, int seed = 42)
{
    srand(seed);
#pragma omp parallel for
    for (int i = 0; i < C; i++)
        for (int j = 0; j < s; j++)
        {
            signal[0 + j * 2 + i * s * 2] = 0.1f * (rand() % 10);
            signal[1 + j * 2 + i * s * 2] = 0.1f * (rand() % 10);
        }
#pragma omp parallel for
    for (int k = 0; k < out; k++)
    {
        for (int i = 0; i < C; i++)
        {
            for (int j = 0; j < f; j++)
            {
                filter[0 + j * 2 + i * f * 2+ k*f*C*2] = 0.1f * (rand() % 10);
                filter[1 + j * 2 + i * f * 2+ k*f*C*2] = 0.1f * (rand() % 10);
            }
        }
    }
}

void doubleToFloat(double* src, float* dest, int size) {
    for (int i = 0; i < size; ++i) {
        dest[i] = static_cast<float>(src[i]);
    }
}

void floatToDouble(float* src, double* dest, int size) {
    if (src == nullptr || dest == nullptr) {
        // handle the error
        std::cerr << "Null pointer passed to floatToDouble" << std::endl;
        return;
    }
    for (int i = 0; i < size; ++i) {
        dest[i] = static_cast<double>(src[i]);
    }
}

void test_speed(double *signal, double *filter,int s, int f, int c, int out)
{
    INITTIMER

    int output_tile_size=s - f + 1;
    fft_conv2d_half *conv2d_lyer;
    double *output_speed = (double *)malloc(sizeof(double) * output_tile_size * output_tile_size * out);

    for (int iter = 0; iter < 5; iter++)
    {
        conv2d_lyer = new fft_conv2d_half(signal, filter, c, out, s, s, f, f, __AMPERE__);
        START
        conv2d_lyer->conv2d_forward(output_speed);
        END_wo_print;
        std::cout << "our_fft_conv"
                << " s " << s << " f " << f << " C " << c << " F " << out << " " << milliseconds << std::endl;
        delete conv2d_lyer;
    }
    for (int iter = 0; iter < 5; iter++)
    {
        conv2d_lyer = new fft_conv2d_half(signal, filter, c, out, s, s, f, f, __AMPERE__);
        START
        conv2d_lyer->conv2d_forward(output_speed);
        END_wo_print;
        std::cout << "our_fft_conv"
                << " s " << s << " f " << f << " C " << c << " F " << out << " " << milliseconds << std::endl;
        delete conv2d_lyer;
    }
}

/*STANDARD relative error func*/
double get_error_2d(double *tested, double *standard, int nx, int ny, int n_batch, int scale = 1)
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

/*STANDARD relative error func*/
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

/*STANDARD CPU DIRECT CONVOLUTION, only use for accuracy check*/
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

void naive_fft_conv2d_fftSize_kernelCount(double *signal, double *filter, double *output, int fftSize, int kernelCount, int sy, int sx, int fy, int fx)
{
    int output_size_x = sx - fx + 1;
    int output_size_y = sy - fy + 1;

    for (int i = 0; i < output_size_y; i++)
    {
        for (int j = 0; j < output_size_x; j++)
        {
            for (int k = 0; k < kernelCount; k++)
            {
                for (int m = 0; m < fy; m++)
                {
                    for (int n = 0; n < fx; n++)
                    {
                        output[(k * output_size_y + i) * output_size_x + j] += filter[(k * fy + m) * fx + n] * signal[(k * sy + i + m) * sx + j + n];
                    }
                }
            }
        }
    }
}


void accuracy(double *signal, double *filter, int s, int f, int C, int kernelCount,int fftSize)
{
    int output_tile_size=s - f + 1;
    double *output_standard = (double *)malloc(sizeof(double) *output_tile_size * output_tile_size * kernelCount);
    double *output_conv = (double *)malloc(sizeof(double) * output_tile_size * output_tile_size * kernelCount);
    float *direct_gpu_results = (float*)malloc(sizeof(float) * fftSize * kernelCount);
    float *turbo_v2_gpu_results = (float*)malloc(sizeof(float) * fftSize * kernelCount);
    float *signal_f = (float *)malloc(sizeof(float) * s * s * C);
    float *filter_f = (float *)malloc(sizeof(float) * f * f * kernelCount * C);
    
    /*Perform standard direct convolution on CPU*/
    naive_fft_conv2d_ss_sf_valid(signal, filter, output_standard, s, s, f, f);

    /*Perform our fft convolution on GPU*/
    fft_conv2d_half conv2d_lyer(signal, filter, C, kernelCount, s, s, f, f, __AMPERE__);
    conv2d_lyer.conv2d_forward(output_conv);

    /*Get accuracy*/
    double acc_conv = get_error_2d(output_conv, output_standard, output_tile_size, output_tile_size, 1);
    std::cout << "our_fft_half_conv VS double_conv"
              << " s " << s << " f " << f << " C " << C << " F " << kernelCount << " relative_error " << acc_conv << std::endl;

    memset(direct_gpu_results, 0, sizeof(float) * fftSize * kernelCount);
    memset(turbo_v2_gpu_results, 0, sizeof(float) * fftSize * kernelCount);
    doubleToFloat(signal,signal_f,s * s * C);
    doubleToFloat(filter,filter_f,f * f * kernelCount * C);

    convolve_direct_fft(signal_f, signal_f, kernelCount, output_tile_size, output_tile_size, f, f, direct_gpu_results);
    convolve_turbo_fft_v2(signal_f, signal_f, kernelCount, output_tile_size, output_tile_size, f, f, turbo_v2_gpu_results);
    double acc_turbo = get_error_f(turbo_v2_gpu_results, direct_gpu_results, fftSize, kernelCount, 1);
    std::cout << "convolve_turbo_fft_v2"
              << " s " << s << " f " << f << " C " << C << " F " << kernelCount << " relative_error " << acc_turbo << std::endl;

}

int main(int argc, char *argv[])
{
    int kernelH = 13;
    int kernelW = 13;
    int dataH = 512 - kernelH + 1;
    int dataW = 512 - kernelW + 1;
    int kernelCount = 16;
    int fftH = dataH + kernelH - 1;
    int fftW = dataW + kernelW - 1;
    int fftSize = fftH * fftW;

    int s = 256,C=1;

    /*Generate input data*/
    double *signal = (double *)malloc(sizeof(double) * s * s * C);
    double *filter = (double *)malloc(sizeof(double) * kernelH * kernelH * kernelCount * C);

    generate_data(signal, filter, s, kernelH, C, kernelCount);

    accuracy(signal, filter,s,kernelH,C,kernelCount,fftSize);
    test_speed(signal, filter,s,kernelH,C,kernelCount);
    return 0;
}