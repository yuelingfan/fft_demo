#pragma once
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <string>

/// @brief Specify your volta gpu(v100) or ampere gpu(A100, A5000,
/// etc.)(default).
enum __GPU_ARCH__
{
    __VOLTA__,
    __AMPERE__
};

/// @brief Specify your fft object(signal or filter), signal as default.
enum __OBJ__
{
    __SIGNAL__,
    __FILTER__
};

/// @brief fft = 1 -> fft, fft = -1 -> ifft
struct fftPlan
{
    __GPU_ARCH__ GPU = __AMPERE__;
    __OBJ__ OBJ = __SIGNAL__;
    int fft;
    int Nx, Ny, N_batch;
    int radices_x[3] = {16, 16, 2};
    int radices_y[3] = {16, 16, 2};
    int n_radices_x, n_radices_y;
    int mergings[2] = {0, 0};
    void (*layer_0[3])(half2 *, half *, half *, int fft);
    void (*layer_1[3])(int, half2 *, half *, half *, int fft);
    half *F_real, *F_imag;
    half *F_real_tmp, *F_imag_tmp;
};
class fft_conv2d_half
{
private:
    /* data */
public:
    /*FOR ALL USE*/
    std::string workpath;
    __GPU_ARCH__ GPU;
    // int num_streams;

    int sx;
    int sy;
    int fx;
    int fy;
    int S;
    int F;
    int padded_x;
    int padded_y;
    int output_x;
    int output_y;

    double *signal;
    double *filter;
    double *cu_signal;
    double *cu_filter;

    half *cu_signal_padded;
    half *cu_filter_padded;

    int *rev_x_cu;
    int *rev_y_cu;
    int rev_256[1024] = {
        0,  16, 32, 48, 64, 80, 96,  112, 128, 144, 160, 176, 192, 208, 224, 240, 1,  17, 33, 49, 65, 81, 97,  113, 129, 145, 161, 177, 193, 209, 225, 241, 2,  18, 34, 50, 66, 82, 98,  114, 130, 146, 162, 178, 194, 210, 226, 242,
        3,  19, 35, 51, 67, 83, 99,  115, 131, 147, 163, 179, 195, 211, 227, 243, 4,  20, 36, 52, 68, 84, 100, 116, 132, 148, 164, 180, 196, 212, 228, 244, 5,  21, 37, 53, 69, 85, 101, 117, 133, 149, 165, 181, 197, 213, 229, 245,
        6,  22, 38, 54, 70, 86, 102, 118, 134, 150, 166, 182, 198, 214, 230, 246, 7,  23, 39, 55, 71, 87, 103, 119, 135, 151, 167, 183, 199, 215, 231, 247, 8,  24, 40, 56, 72, 88, 104, 120, 136, 152, 168, 184, 200, 216, 232, 248,
        9,  25, 41, 57, 73, 89, 105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 10, 26, 42, 58, 74, 90, 106, 122, 138, 154, 170, 186, 202, 218, 234, 250, 11, 27, 43, 59, 75, 91, 107, 123, 139, 155, 171, 187, 203, 219, 235, 251,
        12, 28, 44, 60, 76, 92, 108, 124, 140, 156, 172, 188, 204, 220, 236, 252, 13, 29, 45, 61, 77, 93, 109, 125, 141, 157, 173, 189, 205, 221, 237, 253, 14, 30, 46, 62, 78, 94, 110, 126, 142, 158, 174, 190, 206, 222, 238, 254,
        15, 31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223, 239, 255, 0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0};

    int rev_512[1024] = {0,   32,  64,  96,  128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 2,   34,  66,  98,  130, 162, 194, 226, 258, 290, 322, 354, 386, 418, 450, 482, 4,   36,  68,  100, 132, 164, 196, 228, 260,
                         292, 324, 356, 388, 420, 452, 484, 6,   38,  70,  102, 134, 166, 198, 230, 262, 294, 326, 358, 390, 422, 454, 486, 8,   40,  72,  104, 136, 168, 200, 232, 264, 296, 328, 360, 392, 424, 456, 488, 10,  42,
                         74,  106, 138, 170, 202, 234, 266, 298, 330, 362, 394, 426, 458, 490, 12,  44,  76,  108, 140, 172, 204, 236, 268, 300, 332, 364, 396, 428, 460, 492, 14,  46,  78,  110, 142, 174, 206, 238, 270, 302, 334,
                         366, 398, 430, 462, 494, 16,  48,  80,  112, 144, 176, 208, 240, 272, 304, 336, 368, 400, 432, 464, 496, 18,  50,  82,  114, 146, 178, 210, 242, 274, 306, 338, 370, 402, 434, 466, 498, 20,  52,  84,  116,
                         148, 180, 212, 244, 276, 308, 340, 372, 404, 436, 468, 500, 22,  54,  86,  118, 150, 182, 214, 246, 278, 310, 342, 374, 406, 438, 470, 502, 24,  56,  88,  120, 152, 184, 216, 248, 280, 312, 344, 376, 408,
                         440, 472, 504, 26,  58,  90,  122, 154, 186, 218, 250, 282, 314, 346, 378, 410, 442, 474, 506, 28,  60,  92,  124, 156, 188, 220, 252, 284, 316, 348, 380, 412, 444, 476, 508, 30,  62,  94,  126, 158, 190,
                         222, 254, 286, 318, 350, 382, 414, 446, 478, 510, 1,   33,  65,  97,  129, 161, 193, 225, 257, 289, 321, 353, 385, 417, 449, 481, 3,   35,  67,  99,  131, 163, 195, 227, 259, 291, 323, 355, 387, 419, 451,
                         483, 5,   37,  69,  101, 133, 165, 197, 229, 261, 293, 325, 357, 389, 421, 453, 485, 7,   39,  71,  103, 135, 167, 199, 231, 263, 295, 327, 359, 391, 423, 455, 487, 9,   41,  73,  105, 137, 169, 201, 233,
                         265, 297, 329, 361, 393, 425, 457, 489, 11,  43,  75,  107, 139, 171, 203, 235, 267, 299, 331, 363, 395, 427, 459, 491, 13,  45,  77,  109, 141, 173, 205, 237, 269, 301, 333, 365, 397, 429, 461, 493, 15,
                         47,  79,  111, 143, 175, 207, 239, 271, 303, 335, 367, 399, 431, 463, 495, 17,  49,  81,  113, 145, 177, 209, 241, 273, 305, 337, 369, 401, 433, 465, 497, 19,  51,  83,  115, 147, 179, 211, 243, 275, 307,
                         339, 371, 403, 435, 467, 499, 21,  53,  85,  117, 149, 181, 213, 245, 277, 309, 341, 373, 405, 437, 469, 501, 23,  55,  87,  119, 151, 183, 215, 247, 279, 311, 343, 375, 407, 439, 471, 503, 25,  57,  89,
                         121, 153, 185, 217, 249, 281, 313, 345, 377, 409, 441, 473, 505, 27,  59,  91,  123, 155, 187, 219, 251, 283, 315, 347, 379, 411, 443, 475, 507, 29,  61,  93,  125, 157, 189, 221, 253, 285, 317, 349, 381,
                         413, 445, 477, 509, 31,  63,  95,  127, 159, 191, 223, 255, 287, 319, 351, 383, 415, 447, 479, 511, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};
    void print_cu(half *src, int x, int y, int item_per_group, int batch);
    void print_cu(float *src, int x, int y, int item_per_group, int batch);
    void print_cu(double *src, int x, int y, int item_per_group, int batch);
    void printplan(fftPlan plan);

    void rev(double *src, half *dst, int x, int y, int batch, int offset, cudaStream_t stream);
    void TensorCoreFFTCreate(fftPlan *plan, int nx, int ny, int n_batch, int fft, __GPU_ARCH__ GPU);
    void TensorCoreFFTExec(fftPlan plan, half *data, cudaStream_t stream);
    void TensorCoreFFTExec(fftPlan plan, half *data);
    void gen_rev(int Nx, int Ny);
    void read_out_scale_half2double(half *src, double *dst, double *dst_in_host, int src_x, int src_y, int dst_x, int dst_y, int batch, double scale);
    void read_out_scale_double2double(double *src, double *dst, double *dst_in_host, int src_x, int src_y, int dst_x, int dst_y, int batch, double scale = 1.0);

    void half2double(half *src, double *dst, int x, int y, int item_per_group, int batch, cudaStream_t stream);
    void double2half(double *src, half *dst, int x, int y, int item_per_group, int batch);

    void fft_2d_c2c(double *input, double *output, int x, int y, int batch, __GPU_ARCH__ GPU, int num_streams = 8, bool benchmark = false);
    void ifft_2d_c2c(double *input, double *output, int x, int y, int batch, __GPU_ARCH__ GPU, int num_streams = 8, bool benchmark = false);

    fft_conv2d_half();
    fft_conv2d_half(int in_channel, int out_channel, int kernel_size, __GPU_ARCH__ GPU);
    fft_conv2d_half(double *signal, double *filter, int S, int F, int sy, int sx, int fy, int fx, __GPU_ARCH__ GPU);
    ~fft_conv2d_half();

    void tmp_test();
    std::string toString();
    /*FOR CONV2D FORWARD-PROPAGATION*/

    double *output;

    fftPlan plan_signal;
    fftPlan plan_filter;
    fftPlan plan_pointwise;

    float *cu_pointwise_float;
    half *cu_pointwise_half;
    float *max_per_line;
    float *max_per_tile;

    double *cu_conv2d_forward_result_padded;

    void fft_conv2d_inv_rev_pad();
    void conv2d_forward(double *output);
    void TENSORCORE_FFT_CONV_2D();
    void readout_fwd(bool free = true);
    void fft_pointwise(float scale = 1.0, bool use_more_G_memory = true);
    void fft_conv_final_stage();

    void conv2d_forward(double *input, double *weight, double *output, int S, int F, int sy, int sx, int fy, int fx, __GPU_ARCH__ GPU);

    /*FOR CONV2D BACKWARD-PROPAGATION*/

    fftPlan plan_delta_in_for_bp_filter;
    fftPlan plan_signal_for_bp_filter;
    fftPlan plan_pointwise_for_bp_filter;

    fftPlan plan_delta_in_for_bp_signal;
    fftPlan plan_filter_for_bp_signal;
    fftPlan plan_pointwise_for_bp_signal;

    double *cu_delta_in;
    half *cu_delta_in_padded;
    half *cu_pointwise_for_delta_filter;
    half *cu_pointwise_for_delta_signal;

    double *delta_weight_double;

    double *delta_for_filter;
    double *delta_for_signal;

    void fft_conv2d_rev_pad(double *siganl, double *filter, half *signal_padded, half *filter_padded, int padded_x, int sx, int fx, int S, int F);
    void inv_pad_deltain_signal(int padded_x);
    void pad_fx_1_circles_of_0(double *cu_delta_in_0_padded, int k, int x, int y);
    void pointwise_for_delta_weight(int padded_x);
    void pointwise_for_delta_input(int padded_x, half *delta_for_input_pointwise);
    void conv2d_backward(double *delta_in);
};