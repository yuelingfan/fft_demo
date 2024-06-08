#ifndef FFTCONVOLVE2D_CUH
#define FFTCONVOLVE2D_CUH
#include <cufft.h>

void convolve_direct_fft(float *, float *, int, 
                         int, int, int, int, float *);

void convolve_turbo_fft_v2(float *, float *, int, 
                         int, int, int, int, float *);
void convolve_turbo_fft_v3(float *, float *, int, 
                         int, int, int, int, float *);
void direct_fft(float*, int, int, int, int, int, cufftComplex*);
void qd_fft(float*, int, int, int, int, int, cufftComplex*);
void fast_fft(cufftComplex*, int, int, int, int, int, cufftComplex*, bool);
#endif
