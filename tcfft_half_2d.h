#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

struct tcfftHandle_2d
{
    int Nx, Ny, N_batch;
    int radices_x[3] = {16, 16, 2};
    int radices_y[3] = {16, 16, 2};
    int n_radices_x, n_radices_y;
    int mergings[2] = {0, 0};
    void (*layer_0[3])(half2 *, half *, half *);
    void (*layer_1[3])(int, half2 *, half *, half *);
    half *F_real, *F_imag;
    half *F_real_tmp, *F_imag_tmp;
};

void tcfftExec_2d(tcfftHandle_2d plan, half *data);
void tcfftCreate_2d(tcfftHandle_2d *plan, int nx, int ny, int n_batch);
void setup_2d(double *data, int nx, int ny, int n_batch);
void doit_2d(int iter);
void finalize_2d(double *result,int N_batch);
