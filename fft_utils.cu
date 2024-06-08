#ifndef UTIL_CUH
#define UTIL_CUH

#define IMUL(a, b) __mul24(a, b)
#define FULL_MASK 0xffffffff
//#define IMUL(a, b) a * b
/*
 * Device Code
 */
#include <cufft.h>

////////////////////////////////////////////////////////////////////////////////
// Pad data with zeros, 
////////////////////////////////////////////////////////////////////////////////

int iDivUp(int a, int b){
    return (a % b != 0) ? ( a / b + 1) : ( a / b );
}

__global__ void padDataWithZeros(
    float *d_PaddedData,
    const float *d_Data,
    int fftW,
    int fftH,
    int dataW,
    int dataH,
    int FEATURE_DIM
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    const int z = IMUL(blockDim.z, blockIdx.z) + threadIdx.z;

    if(x < fftW && y < fftH && z < FEATURE_DIM){
        if(x < dataW && y < dataH)
            //d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(x, fftH) + y] = 
            //        d_Data[ IMUL(z, IMUL(dataH, dataW)) + IMUL(x, dataH ) + y];
            //d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(y, fftW) + x] = 
            //        d_Data[ IMUL(z, IMUL(dataH, dataW)) + IMUL(y, dataW ) + x];
            d_PaddedData[z * fftW * fftH + y * fftW + x] = 
                    d_Data[z * dataH * dataW + y * dataW + x];
        //else
            //d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(x, fftH) + y] = 0;
        //    d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(y, fftW) + x] = 0;
    }
}


__global__ void padDataWithZerosComplex(
    float2 *d_PaddedData,
    const float *d_Data,
    int fftW,
    int fftH,
    int dataW,
    int dataH,
    int FEATURE_DIM
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    const int z = IMUL(blockDim.z, blockIdx.z) + threadIdx.z;

    if(x < fftW && y < fftH && z < FEATURE_DIM ){
        if(x < dataW && y < dataH)
            //d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(x, fftH) + y] = 
            //        d_Data[ IMUL(z, IMUL(dataH, dataW)) + IMUL(x, dataH ) + y];
            //d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(y, fftW) + x] = 
            //        d_Data[ IMUL(z, IMUL(dataH, dataW)) + IMUL(y, dataW ) + x];
            d_PaddedData[z * fftW * fftH + y * fftW + x].x = 
                    d_Data[z * dataH * dataW + y * dataW + x];
        //else
            //d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(x, fftH) + y] = 0;
        //    d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(y, fftW) + x] = 0;
    }
}

__global__ void padKernelWithZerosComplex(
    float2 *d_PaddedData,
    const float *d_Data,
    int fftW,
    int fftH,
    int dataW,
    int dataH,
    int GROUP_NUM
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    const int z = IMUL(blockDim.z, blockIdx.z) + threadIdx.z;
    //const int x = threadIdx.x;
    //const int y = threadIdx.y;
    //const int z = IMUL(blockDim.z, blockIdx.x) + threadIdx.z;

    //if(x < fftW && y < fftH && z < GROUP_NUM ){
        if(x < dataW && y < dataH && z < GROUP_NUM){
            //d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(x, fftH) + y] = 
            //        d_Data[ IMUL(z, IMUL(dataH, dataW)) + IMUL(x, dataH ) + y];
            //d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(y, fftW) + x] = 
            //        d_Data[ IMUL(z, IMUL(dataH, dataW)) + IMUL(y, dataW ) + x];
            d_PaddedData[z * fftW * fftH + y * fftW + x].x = 
                    d_Data[z * 2 * dataH * dataW + y * dataW + x];
            d_PaddedData[z * fftW * fftH + y * fftW + x].y = 
                    d_Data[(z * 2 + 1) * dataH * dataW + y * dataW + x];
        //else
            //d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(x, fftH) + y] = 0;
        //    d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(y, fftW) + x] = 0;
        }
    //}
}

__global__ void ConjComplex(
    float2 *d_conj,
    const float2 *d_original,
    int dataW,
    int dataH,
    int GROUP_NUM
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    const int z = IMUL(blockDim.z, blockIdx.z) + threadIdx.z;

    if(x < dataW && y < dataH && z < GROUP_NUM){
            //d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(x, fftH) + y] = 
            //        d_Data[ IMUL(z, IMUL(dataH, dataW)) + IMUL(x, dataH ) + y];
            //d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(y, fftW) + x] = 
            //        d_Data[ IMUL(z, IMUL(dataH, dataW)) + IMUL(y, dataW ) + x];
            d_conj[z * dataW * dataH + y * dataW + x].x = 
                    d_original[z * 2 * dataH * dataW + y * dataW + x].x;
            d_conj[z * dataW * dataH + y * dataW + x].y = 
                    - d_original[z * 2 * dataH * dataW + y * dataW + x].y;
        //else
            //d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(x, fftH) + y] = 0;
        //    d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(y, fftW) + x] = 0;
    }
}

__global__ void SplitComplex(
    float *d_results,
    const float2 *d_original,
    int dataW,
    int dataH,
    int GROUP_NUM
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    const int z = IMUL(blockDim.z, blockIdx.z) + threadIdx.z;

    float2 local_data;
    if(x < dataW && y < dataH && z < GROUP_NUM){
            //d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(x, fftH) + y] = 
            //        d_Data[ IMUL(z, IMUL(dataH, dataW)) + IMUL(x, dataH ) + y];
            //d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(y, fftW) + x] = 
            //        d_Data[ IMUL(z, IMUL(dataH, dataW)) + IMUL(y, dataW ) + x];
	    local_data = d_original[z * dataH * dataW + y * dataW + x];
            d_results[z * 2 * dataW * dataH + y * dataW + x] = 
                    local_data.x;
            d_results[(z * 2 + 1) * dataW * dataH + y * dataW + x] = 
                    local_data.y;
        //else
            //d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(x, fftH) + y] = 0;
        //    d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(y, fftW) + x] = 0;
    }
}
__global__ void qdVectorConstruct(
	const cufftComplex *Data,
	const int* input_xs,
	const int* input_ys,
	const int N,
	int group_num,
	int qdV_num,
	cufftComplex *qdVectors
	)
{

	const int group_id = blockIdx.x;
	const int qdV_id = blockIdx.y;

	const int group_offset = group_id * qdV_num * N;
	const int qdV_offset = qdV_id * N;
	const int elem_offset = blockDim.x * threadIdx.y + threadIdx.x;
	const int lane_id = threadIdx.x;

	const int global_offset = group_offset + qdV_offset + elem_offset;
	float local_real = 0;
	float local_img = 0;

        //qdVectors[global_offset].x = global_offset;
	//qdVectors[global_offset].y = blockIdx.y;
	//return;

	__shared__ float warp_reals[256];
	__shared__ float warp_imgs[256];

	warp_reals[lane_id] = 0;
	warp_imgs[lane_id] = 0;
	for(int t = 0; t < 32; t++){
	    local_real = 0;
	    local_img = 0;
	    for(int j = 0; j < N; j+=32){
	        int index_offset = qdV_id * N * N + (blockDim.x * threadIdx.y + t) * N + j + lane_id;
	        int idx_x = input_xs[index_offset];
	        int idx_y = input_ys[index_offset];

		float tmp_real = Data[group_id * N * N + idx_x * N + idx_y].x;
		float tmp_img = Data[group_id * N * N + idx_x * N + idx_y].y;

	        // in-warp reduce
		for (int offset = 16; offset > 0; offset /= 2){
		    tmp_real += __shfl_down_sync(FULL_MASK, tmp_real, offset);
		    tmp_img += __shfl_down_sync(FULL_MASK, tmp_img, offset);
		}
		if(lane_id == 0){
		    local_real += tmp_real;
		    local_img += tmp_img;
		}
	    }
	    if (lane_id == 0){
		warp_reals[blockDim.x * threadIdx.y + t] = local_real;
		warp_imgs[blockDim.x * threadIdx.y + t] = local_img;
	    }
	}
	qdVectors[global_offset].x = warp_reals[elem_offset];
	qdVectors[global_offset].y = warp_imgs[elem_offset];
}

__global__ void qdVectorReconstruct(
	const cufftComplex *w_Data,
	const int* output_xs,
	const int* output_ys,
	const int N,
	int group_num,
	cufftComplex *results
	)
{

	const int group_id = blockIdx.x;
	const int qdV_id = blockIdx.y * 32 + threadIdx.x;

        //qdVectors[global_offset].x = global_offset;
	//qdVectors[global_offset].y = blockIdx.y;
	//return;

	int output_idx = output_xs[qdV_id];
	int output_idy = output_ys[qdV_id];
	results[group_id * N * N + output_idx * N + output_idy].x = w_Data[group_id * N * N * 3 / 2 + qdV_id ].x;
	results[group_id * N * N + output_idx * N + output_idy].y = w_Data[group_id * N * N * 3 / 2 + qdV_id ].y;

}

////////////////////////////////////////////////////////////////////////////////
// Copy input data array to the upper left corner and pad by border values
////////////////////////////////////////////////////////////////////////////////
//texture<float, 3, cudaReadModeElementType> texData;

__global__ void padDataWithBorder(
	float *d_PaddedData,
        float *d_Data,
	int fftW,
	int fftH,
	int dataW,
	int dataH,
	int featureDim,
	int kernelW,
	int kernelH,
	int kernelX,
	int kernelY
){
	const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
	const int borderW = dataW + kernelX;
	const int borderH = dataH + kernelY;
	int dx;
	int dy;

	if(x < fftW && y < fftH){
		if(x < dataW) dx = x;
		if(y < dataH) dy = y;
		if(x >= dataW && x < borderW) dx = dataW - 1;
		if(y >= dataH && y < borderH) dy = dataH - 1;
		if(x >= borderW) dx = 0;
		if(y >= borderH) dy = 0;

		// d_PaddedData[IMUL(y, fftW) + x] =
		// 	tex2D(texData, (float)dx + 0.5f, (float)dy + 0.5f);
		d_PaddedData[IMUL(y, fftW) + x] = d_Data[IMUL(dy, dataW) + dx];
	}
}


////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
__device__ void complexMulAndScale(cufftComplex &out, cufftComplex a, cufftComplex b, float c){
    const cufftComplex t = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};
    out = t;
}

__device__ void complexConjMulAndScale(cufftComplex &out, cufftComplex a, cufftComplex b, float c){
    const cufftComplex t = {c * (a.x * b.x + a.y * b.y), c * (a.y * b.x - a.x * b.y)};
    out = t;
}

__global__ void elementwiseProductAndNormalize(
    cufftComplex *fft_Output,
    const cufftComplex *fft_PaddedData,
    const cufftComplex *fft_PaddedKernel,
    int FFT_H,
    int FFT_W,
    int FEATURE_DIM,
    float scale
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    const int z = IMUL(blockDim.z, blockIdx.z) + threadIdx.z;
    
    if(x < FFT_W && y < FFT_H && z < FEATURE_DIM){
        // int i = IMUL(z, IMUL(FFT_W, FFT_H)) + IMUL(FFT_H, x) + y;
        int i = z * FFT_W * FFT_H + FFT_H * x + y;
        // complexConjMulAndScale(fft_Output[i], fft_PaddedData[i], fft_PaddedKernel[i], scale);
        fft_Output[i].x = scale * (fft_PaddedData[i].x * fft_PaddedKernel[i].x - fft_PaddedData[i].y * fft_PaddedKernel[i].y);
        fft_Output[i].y = scale * (fft_PaddedData[i].y * fft_PaddedKernel[i].x + fft_PaddedData[i].x * fft_PaddedKernel[i].y);
    }
}

/* Support in-place computation, i.e. input and output can be the same */
__global__ void sumAlongFeatures(
    float *convolutionResult,
    const float *convolutionPerFeature,
    int FFT_H,
    int FFT_W,
    int FEATURE_DIM
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;

    if(x < FFT_W && y < FFT_H){
        const int result_i = IMUL(FFT_H, x) + y;
        const int N = IMUL(FFT_W, FFT_H);

        float acc = convolutionPerFeature[result_i];
        int zN = N;
        for (int z = 1; z < FEATURE_DIM; z++){
            acc += convolutionPerFeature[zN + result_i];
            zN += N;
        }
        convolutionResult[result_i] = acc;
    }
}
    
__global__ void qdVectorConstruct_sparse(
	const cufftComplex *Data,
	const int* sparse_input_xs,
	const int* sparse_input_ys,
	const int* begin_arr,
	const int* end_arr,
	const int* qdV_arr,
	const int N,
        const int k, 
	int group_num,
	int non_empty_qdV,
	cufftComplex *qdVectors
	)
{
	const int lane_id = threadIdx.x;
	const int warp_id = blockIdx.y * blockDim.y + threadIdx.y;
	const int group_id = blockIdx.x;
	const int qdV_id = blockIdx.y * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	const int blockSize = 32;
        int kN = k;
	//const int blockSize = 256;

	//__shared__ float kernel_reals[kN * kN];
	//__shared__ float kernel_imgs[kN * kN];
	__syncthreads();
	// last warp
	if((warp_id + 1) * 32 > non_empty_qdV){
	    if (k == 3){
	        __shared__ float2 kernels[9];
	        for (int i = 0 ; i < kN * kN; i += blockSize)
	        {
	            int cur_ki = i + blockDim.x * threadIdx.y + lane_id;
	            if(cur_ki < kN * kN){
	                int kx = cur_ki % kN;
	                int ky = cur_ki / kN;
	                //kernel_reals[cur_ki] = Data[group_id * N * N + ky * N + kx].x;
	                //kernel_imgs[cur_ki] = Data[group_id * N * N + ky * N + kx].y;
	        	kernels[cur_ki] = Data[group_id * N * N + ky * N + kx];
	            }
	        }
	        if(qdV_id < non_empty_qdV){
                    int begin_idx = begin_arr[qdV_id];
	            int end_idx = end_arr[qdV_id];
	            int non_qdV_idx = qdV_arr[qdV_id];

	            //float local_real = 0;
	            //float local_img = 0;
	            float2 local_val;
	            local_val.x = 0;
	            local_val.y = 0;
	            for(int s = begin_idx; s < end_idx; s++){
	                int idx_x = sparse_input_xs[s];
	                int idx_y = sparse_input_ys[s];
	                //local_real += kernel_reals[idx_x * kN + idx_y];
	                //local_img  += kernel_imgs[idx_x * kN + idx_y];
	                local_val.x += kernels[idx_x * kN + idx_y].x;
	                local_val.y  += kernels[idx_x * kN + idx_y].y;
	            
	            }

	            //qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx].x = local_real;
	            //qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx].y = local_img;
	            qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx] = local_val;
	        
	        }
	    }
            else if(k == 5){
                __shared__ float2 kernels[25];
	        for (int i = 0 ; i < kN * kN; i += blockSize)
	        {
	            int cur_ki = i + blockDim.x * threadIdx.y + lane_id;
	            if(cur_ki < kN * kN){
	                int kx = cur_ki % kN;
	                int ky = cur_ki / kN;
	                //kernel_reals[cur_ki] = Data[group_id * N * N + ky * N + kx].x;
	                //kernel_imgs[cur_ki] = Data[group_id * N * N + ky * N + kx].y;
	        	kernels[cur_ki] = Data[group_id * N * N + ky * N + kx];
	            }
	        }
	        if(qdV_id < non_empty_qdV){
                    int begin_idx = begin_arr[qdV_id];
	            int end_idx = end_arr[qdV_id];
	            int non_qdV_idx = qdV_arr[qdV_id];

	            //float local_real = 0;
	            //float local_img = 0;
	            float2 local_val;
	            local_val.x = 0;
	            local_val.y = 0;
	            for(int s = begin_idx; s < end_idx; s++){
	                int idx_x = sparse_input_xs[s];
	                int idx_y = sparse_input_ys[s];
	                //local_real += kernel_reals[idx_x * kN + idx_y];
	                //local_img  += kernel_imgs[idx_x * kN + idx_y];
	                local_val.x += kernels[idx_x * kN + idx_y].x;
	                local_val.y  += kernels[idx_x * kN + idx_y].y;
	            
	            }

	            //qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx].x = local_real;
	            //qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx].y = local_img;
	            qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx] = local_val;
	        
	        }
	    }
	
	}
	// other warps
	else
	{
            int begin_idx = begin_arr[qdV_id];
	    int end_idx = end_arr[qdV_id];
	    int non_qdV_idx = qdV_arr[qdV_id];
	    int global_begin_idx = 0;
	    int global_end_idx = 0;

            if (kN == 3){
	        __shared__ float2 kernels[9];
	        __shared__ int input_xs[blockSize * 3];
	        __shared__ int input_ys[blockSize * 3];
	        for (int i = 0 ; i < kN * kN; i += blockSize)
	        {
	            int cur_ki = i + blockDim.x * threadIdx.y + lane_id;
	            if(cur_ki < kN * kN){
	                int kx = cur_ki % kN;
	                int ky = cur_ki / kN;
	                //kernel_reals[cur_ki] = Data[group_id * N * N + ky * N + kx].x;
	                //kernel_imgs[cur_ki] = Data[group_id * N * N + ky * N + kx].y;
	        	kernels[cur_ki] = Data[group_id * N * N + ky * N + kx];
	            }
	        }
	        // read all the input xs and ys
	        global_begin_idx = __shfl_sync(FULL_MASK, begin_idx, 0, 32);
	        global_end_idx = __shfl_sync(FULL_MASK, end_idx, 31, 32);

	        //__syncthreads();
	        //if (blockDim.x * threadIdx.y + lane_id == 0)
	        //    global_begin_idx = begin_idx;
	        //if (blockDim.x * threadIdx.y + lane_id == 255)
	        //    global_end_idx = end_idx;
	        //__syncthreads();

	        //qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx].x = global_begin_idx;
	        //qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx].y = global_end_idx;
	        //return;

	        int non_zero = global_end_idx - global_begin_idx;
	        for(int i = global_begin_idx; i < global_end_idx;i+=blockSize){
	            int cur_idx = i + blockDim.x * threadIdx.y + lane_id;
	            if(cur_idx < global_end_idx){
	        	    input_xs[cur_idx - global_begin_idx] = sparse_input_xs[cur_idx];
	        	    input_ys[cur_idx - global_begin_idx] = sparse_input_ys[cur_idx];
	            }
	        }

	        //float local_real = 0;
	        //float local_img = 0;
	        float2 local_val;
	        local_val.x = 0;
	        local_val.y = 0;
	        for(int s = begin_idx; s < end_idx; s++){
	            int idx_x = input_xs[s - global_begin_idx];
	            int idx_y = input_ys[s - global_begin_idx];
	            //local_real += kernel_reals[idx_x * kN + idx_y];
	            //local_img  += kernel_imgs[idx_x * kN + idx_y];
	            local_val.x += kernels[idx_x * kN + idx_y].x;
	            local_val.y += kernels[idx_x * kN + idx_y].y;
	        
	        }

	        //qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx].x = local_real;
	        //qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx].y = local_img;
	        qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx] = local_val;

	    }	
            else if (kN == 5){
	        __shared__ float2 kernels[25];
	        __shared__ int input_xs[blockSize * 5];
	        __shared__ int input_ys[blockSize * 5];
	        for (int i = 0 ; i < kN * kN; i += blockSize)
	        {
	            int cur_ki = i + blockDim.x * threadIdx.y + lane_id;
	            if(cur_ki < kN * kN){
	                int kx = cur_ki % kN;
	                int ky = cur_ki / kN;
	                //kernel_reals[cur_ki] = Data[group_id * N * N + ky * N + kx].x;
	                //kernel_imgs[cur_ki] = Data[group_id * N * N + ky * N + kx].y;
	        	kernels[cur_ki] = Data[group_id * N * N + ky * N + kx];
	            }
	        }
	        // read all the input xs and ys
	        global_begin_idx = __shfl_sync(FULL_MASK, begin_idx, 0, 32);
	        global_end_idx = __shfl_sync(FULL_MASK, end_idx, 31, 32);

	        //__syncthreads();
	        //if (blockDim.x * threadIdx.y + lane_id == 0)
	        //    global_begin_idx = begin_idx;
	        //if (blockDim.x * threadIdx.y + lane_id == 255)
	        //    global_end_idx = end_idx;
	        //__syncthreads();

	        //qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx].x = global_begin_idx;
	        //qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx].y = global_end_idx;
	        //return;

	        int non_zero = global_end_idx - global_begin_idx;
	        for(int i = global_begin_idx; i < global_end_idx;i+=blockSize){
	            int cur_idx = i + blockDim.x * threadIdx.y + lane_id;
	            if(cur_idx < global_end_idx){
	        	    input_xs[cur_idx - global_begin_idx] = sparse_input_xs[cur_idx];
	        	    input_ys[cur_idx - global_begin_idx] = sparse_input_ys[cur_idx];
	            }
	        }

	        //float local_real = 0;
	        //float local_img = 0;
	        float2 local_val;
	        local_val.x = 0;
	        local_val.y = 0;
	        for(int s = begin_idx; s < end_idx; s++){
	            int idx_x = input_xs[s - global_begin_idx];
	            int idx_y = input_ys[s - global_begin_idx];
	            //local_real += kernel_reals[idx_x * kN + idx_y];
	            //local_img  += kernel_imgs[idx_x * kN + idx_y];
	            local_val.x += kernels[idx_x * kN + idx_y].x;
	            local_val.y += kernels[idx_x * kN + idx_y].y;
	        
	        }

	        //qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx].x = local_real;
	        //qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx].y = local_img;
	        qdVectors[group_id * N * N * 3 / 2 + non_qdV_idx] = local_val;
	    }	
	
	}


}

__global__ void qdVectorReconstruct_Shared(
	const cufftComplex *w_Data,
	const int* output_xs,
	const int* output_ys,
	const int N,
	int group_num,
	cufftComplex *results,
	bool TOP_HALF
	)
{

	const int group_id = blockIdx.x;
	int qdV_id = blockIdx.y * N + threadIdx.x;
    const int rows = 4;
	if(TOP_HALF)
	{
	    qdV_id = threadIdx.x * N + blockIdx.y * rows + threadIdx.y;
	    //qdV_id = N * rows * blockIdx.y + N * threadIdx.y + threadIdx.x;
	}
    //qdVectors[global_offset].x = global_offset;
	//qdVectors[global_offset].y = blockIdx.y;
	//return;

	if(TOP_HALF){
            if(N == 32){
	        __shared__ float2 partial_data[32 * rows];
		int elem_idx = qdV_id;
		//int output_idx = output_xs[elem_idx];
		int output_idy= output_ys[elem_idx];
		partial_data[threadIdx.y * N + output_idy] = w_Data[group_id * N * N * 3 /2 + elem_idx];
	        __syncthreads();
	        results[group_id * N * N + (blockIdx.y * 4 + threadIdx.y) * N + threadIdx.x] = partial_data[threadIdx.y * N + threadIdx.x];
	    }
            else if (N == 64){
                __shared__ float2 partial_data[64 * rows];
		int elem_idx = qdV_id;
		//int output_idx = output_xs[elem_idx];
		int output_idy= output_ys[elem_idx];
		partial_data[threadIdx.y * N + output_idy] = w_Data[group_id * N * N * 3 /2 + elem_idx];
	        __syncthreads();
	        results[group_id * N * N + (blockIdx.y * 4 + threadIdx.y) * N + threadIdx.x] = partial_data[threadIdx.y * N + threadIdx.x];
	    }
	    //for(int j = 0; j < rows;j++)
	    //{
	    //    partial_data[j * N + threadIdx.x].x = 0;
	    //    partial_data[j * N + threadIdx.x].y = 0;
	    //}
	    //__syncthreads();

	    //for(int ti = 0; ti < N / rows; ti++){
		//    for(int tj = 0; tj < rows;tj++){
		//        int elem_idx = qdV_id * N + ti * rows + tj;
		//        //int output_idx = output_xs[elem_idx];
		//        int output_idy = output_ys[elem_idx];
		//        //int shared_idx = output_idx - ti * 4;
		//        partial_data[tj * N + output_idy] = w_Data[group_id * N * N * 3 /2 + elem_idx];

		//    }
	    //    __syncthreads();
	    //    for(int j = 0; j < rows;j++)
	    //        results[group_id * N * N + (ti * rows + j) * N + threadIdx.x] = partial_data[j * N + threadIdx.x];
	    //    __syncthreads();
	    //
	    //}

	}
	else {
	    qdV_id += N;
	    for(int t = 0; t < N;t++){
		int elem_idx = qdV_id * N + t;
	        int output_idx = output_xs[elem_idx];
	        int output_idy = output_ys[elem_idx];
	        results[group_id * N * N + output_idx * N + output_idy].x = w_Data[group_id * N * N * 3 / 2 + elem_idx].x;
	        results[group_id * N * N + output_idx * N + output_idy].y = w_Data[group_id * N * N * 3 / 2 + elem_idx].y;
	    }
	}
}

__global__ void qdVectorReconstruct_Small(
	const cufftComplex *w_Data,
	const int* output_xs,
	const int* output_ys,
	const int N,
	int group_num,
	cufftComplex *results,
	bool TOP_HALF
	)
{

	const int group_id = blockIdx.x;
	//int qdV_id = threadIdx.x;
	int fN = N;
	const int quad_fN = fN / 4;
	//const int qdV_num = 96;
    //qdVectors[global_offset].x = global_offset;
	//qdVectors[global_offset].y = blockIdx.y;
	//return;

	int elem_idx = 0;
	int output_idx = 0;
	int output_idy = 0;
        if (fN == 32){
	    __shared__ float2 partial_data[1024];
	    for(int i = 0; i < 4; i++){
	        elem_idx = (i * quad_fN + threadIdx.y) * fN + threadIdx.x;
	        output_idx = threadIdx.x;
	        output_idy = output_ys[elem_idx];
	        partial_data[output_idx * N + output_idy] = w_Data[group_id * N * N * 3 / 2 + elem_idx];
	    }
	    __syncthreads();
	    //if(threadIdx.y < 32){
	    for(int i = 0; i < 2;i++){
	    	elem_idx = ((i + 4) * quad_fN + threadIdx.y) * N + threadIdx.x;
	    	output_idx = output_xs[elem_idx];
	    	output_idy = output_ys[elem_idx];
	    	partial_data[output_idx * N + output_idy] = w_Data[group_id * N * N * 3 / 2 + elem_idx];
	    }
	    __syncthreads();
	    for (int i = 0; i < 4; i++)
            results[group_id * N * N + (i * quad_fN + threadIdx.y) * fN + threadIdx.x] = partial_data[(i * quad_fN + threadIdx.y) * N + threadIdx.x];
	}
        else if (fN == 64){
	    __shared__ float2 partial_data[4096];
	    for(int i = 0; i < 4; i++){
	        elem_idx = (i * quad_fN + threadIdx.y) * fN + threadIdx.x;
	        output_idx = threadIdx.x;
	        output_idy = output_ys[elem_idx];
	        partial_data[output_idx * N + output_idy] = w_Data[group_id * N * N * 3 / 2 + elem_idx];
	    }
	    __syncthreads();
	    //if(threadIdx.y < 32){
	    for(int i = 0; i < 2;i++){
	    	elem_idx = ((i + 4) * quad_fN + threadIdx.y) * N + threadIdx.x;
	    	output_idx = output_xs[elem_idx];
	    	output_idy = output_ys[elem_idx];
	    	partial_data[output_idx * N + output_idy] = w_Data[group_id * N * N * 3 / 2 + elem_idx];
	    }
	    __syncthreads();
	    for (int i = 0; i < 4; i++)
            results[group_id * N * N + (i * quad_fN + threadIdx.y) * fN + threadIdx.x] = partial_data[(i * quad_fN + threadIdx.y) * N + threadIdx.x];
	}


}

__global__ void dotProduct(
	const cufftComplex *signal_fft,
	const int N,
	const int half_N, 
	const float inverse_fftSize, 
	int group_num,
	cufftComplex *kernels_fft
	)
{

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockIdx.z;

	if(y < half_N){
        //__shared__ float2 patch[32 * 32];
	    //patch[threadIdx.y * 32 + threadIdx.x] = signal_fft[y * N + x];
	    float2 s_value = signal_fft[y * N + x];
	    for(int k = z * group_num; k < (z+1)*group_num;k++){
	        float2 k_value = kernels_fft[k * N * half_N + y * N + x];
	        float2 result;
	        result.x = (s_value.x * k_value.x - s_value.y * k_value.y) * inverse_fftSize;
	        result.y = (s_value.x * k_value.y + s_value.y * k_value.x) * inverse_fftSize;
	    
	        kernels_fft[k * N * half_N + y * N + x] = result;
	    }
    }
}
#endif
