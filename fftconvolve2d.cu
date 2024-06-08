/**
    fftconvolve2d.cu
    Compute real-complex FFT convolutions on the GPU

    @author: Qiang Wang (qiangwang@comp.hkbu.edu.hk)
    @version: 1.0 05/12/2018
*/


#include "cuda_utils.h"
#include "fft_utils.cu"
#include "fftconvolve2d.cuh"
#include <chrono>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <cuda_profiler_api.h>
#include <numeric>


#define DEBUG 0
#define VERBOSE 0
#define QDFFT_MEASURE 1
#define NUM_ITERATIONS 100

using namespace std;

//extern "C" {
    std::unordered_map<std::string, std::vector<double>> timerMap;

    // index array for qdfft
    int* input_xs = NULL;
    int* input_ys = NULL;
    int* output_xs = NULL;
    int* output_ys = NULL;
    int* d_input_xs = NULL;
    int* d_input_ys = NULL;
    int* d_output_xs = NULL;
    int* d_output_ys = NULL;
    int* p = NULL;
    int* s = NULL;

    int* sparse_input_xs = NULL;
    int* sparse_input_ys = NULL;
    int* begin_arr = NULL;
    int* end_arr = NULL;
    int* qdV_arr = NULL;
    int non_zero = 0;
    int non_empty_qdV = 0;

    int* d_sparse_input_xs = NULL;
    int* d_sparse_input_ys = NULL;
    int* d_begin_arr = NULL;
    int* d_end_arr = NULL;
    int* d_qdV_arr = NULL;
    cufftComplex *d_qdVectors = NULL;
    cufftComplex* d_w_results = NULL;
    cufftHandle plan1d; 

    void reset_timer()
    {
        timerMap.clear();
    }

    void report_timer()
    {
        double total_time = 0;
        if (timerMap.find("total_time") != timerMap.end())
            total_time = std::accumulate(timerMap["total_time"].begin() + 1,
                                         timerMap["total_time"].end(), 0.0);
        for (auto &kv : timerMap) {
            auto v = kv.second;
            double sum = std::accumulate(v.begin() + 1, v.end(), 0.0);
            printf("[%s] Calls: %d, Average time: %.3lf ms, Global Percentage: %.2lf %%\n",
                   kv.first.c_str(), (int)v.size(), sum / (int)v.size(), 100.0 * sum / total_time);
        }
    }


    struct complexMultiplier
    {
        double scale;
        complexMultiplier(double scale): scale(scale) {};

        __host__ __device__
        cufftComplex operator() (const cufftComplex &v1,
                                       const cufftComplex &v2) const
        {
            cufftComplex res;
            res.x = (v1.x * v2.x - v1.y * v2.y) * scale;
            res.y = (v1.x * v2.y + v1.y * v2.x) * scale;
            return res;
        }
    };

    struct complexAdder
    {
        double scale;
        complexAdder(double scale): scale(scale) {};

        __host__ __device__
        cufftComplex operator() (const cufftComplex &v1,
                                       const cufftComplex &v2) const
        {
            cufftComplex res;
            res.x = (v1.x + v2.x) * scale;
            res.y = (v1.y + v2.y) * scale;
            return res;
        }
    };

    /**
        Creates an FFT Plan if it has not been yet initialized

        @plan: Pointer to the plan that will be created/initialized
        @size: Size of the FFT for which this plan will be used
        @type: Type of the FFT
        @batch: Number of FFTs of the specified size that will be computed together.

    */
    void create_plan(cufftHandle *plan, size_t nRows, size_t nCols, cufftType type)
    {
        size_t workSize;
        int ret = cufftGetSize(*plan, &workSize);
        if (ret == CUFFT_INVALID_PLAN) {
            if (cufftPlan2d(plan, nRows, nCols, type) != CUFFT_SUCCESS) {
                fprintf(stderr, "CUFFT error: Plan creation failed");
            }
        }
    }

    // helper function for QD fft
    void gcd(int p, int s, int& x, int& y, int& r){
	if(p == 0){
	    x = 0;
	    y = 1;
	    r = s;
	    return;
	}
	if(s == 0){
	    x = 1;
	    y = 0;
	    r = p;
	    return;
	}
	gcd(s, p % s, x, y, r);
	int temp = x;
	x = y;
	y = temp - p / s * y;
	return;
    }

    void find_input_index(int p, int s, int t, int N, int* idx_x, int* idx_y){
	int a0 = 0, b0 = 0, gcd_value = 0;
	gcd(p, s, a0, b0, gcd_value);
	a0 = a0 * t;
	b0 = b0 * t;
	//printf("find gcd: p:%d, s:%d, a:%d, b:%d, gcd:%d.\n", p, s, a0, b0, gcd_value);
	for(int i = 0;i<N;i++){
	    idx_x[i] = (a0 + s / gcd_value * i) % N;
	    idx_y[i] = (b0 - p / gcd_value * i + N * N) % N;
	}
    }


    /**
        Computes the FFT convolution of two padded signals, direct fft

        @signal: The first signal. This is a pointer to host(CPU) memory
        @signalSize: The signal size
        @kernel: The second signal, also called kernel. This is a pointer to
                 host(CPU) memory
        @kernelSize: The kernel size
        @result: Pointer to host(CPU) memory that contains the convolution result.
                 Sufficient memory ((singalSize + kernelSize -1) * sizeof(cufftDoubleComplex))
                 has to be allocated before calling the function.
        @d_in: Pointer to GPU memory used by the function. The size of the memory region
                has to be at least 2 * (signalSize + kernelSize - 1)
        @fwplan: An integer handle used to store the forward FFT plan.
        @bwplan: An integer handle used to store the backward FFT plan.
    */
    void convolve_direct_fft(float * Signal, float * Kernels, int kernelCount, 
                             int nRows, int nCols, int kRows, int kCols, float * results)
    {

        cufftHandle fwplan_signal, fwplan_kernels, bwplan; 
        // timer timer, globalTimer;
        // globalTimer.restart();

        int fftRows = nRows + kRows - 1;
        int fftCols = nCols + kCols - 1;
        int dataSize = nRows * nCols;
        int kernelSize = kRows * kCols;
        int fftSize = fftRows * fftCols;
        int cfftSize = fftRows * (fftCols / 2 + 1);

        // parameters for cufftPlanMany
        int idist = fftRows * fftCols;
        int odist = fftRows * (fftCols / 2 + 1);
        int inembed[] = {fftRows, fftCols};
        int onembed[] = {fftRows, fftCols / 2 + 1};
        int n[2] = {fftRows, fftCols};

#if DEBUG
        for(int i = 0;i<nRows;i++){
            for(int j = 0;j<nCols;j++)
                printf("%f ", Signal[i * nCols + j]);
            printf("\n");
        }
        printf("\n");
        for(int i = 0;i<kRows;i++){
            for(int j = 0;j<kCols;j++)
                printf("%f ", Kernels[i * kCols + j]);
            printf("\n");
	}
        printf("\n");
#endif

        // initialize device memory
        float *d_signal = NULL;   // device storage of input signal
        float *d_kernels = NULL;  // device storage of kernels and final results
        float *d_paddedSignal = NULL;
        float *d_paddedKernels = NULL;  // device storage of kernels and final results
        float *d_results = NULL;  
        cufftComplex *d_signal_fft;
        cufftComplex *d_kernels_fft;
        
        cudaMalloc((void**)&d_signal, sizeof(float) * dataSize);
        cudaMalloc((void**)&d_kernels, sizeof(float) * kernelSize * kernelCount);
        cudaMalloc((void**)&d_paddedSignal, sizeof(float) * fftSize);
        cudaMalloc((void**)&d_paddedKernels, sizeof(float) * fftSize * kernelCount);
        cudaMalloc((void**)&d_results, sizeof(float) * fftSize * kernelCount);
        cudaMalloc((void**)&d_signal_fft, cfftSize * sizeof(cufftComplex));
        cudaMalloc((void**)&d_kernels_fft, cfftSize * kernelCount * sizeof(cufftComplex));

        // timer.restart();
        cudaMemset(d_signal, 0, dataSize * sizeof(float));
        cudaMemset(d_kernels, 0, kernelSize * kernelCount * sizeof(float));
        cudaMemset(d_paddedSignal, 0, fftSize * sizeof(float));
        cudaMemset(d_paddedKernels, 0, fftSize * kernelCount * sizeof(float));
        cudaMemset(d_results, 0, fftSize * kernelCount * sizeof(float));
        //cudaMemset(d_signal_fft, 0, odist * sizeof(cufftComplex));
        //cudaMemset(d_kernels_fft, 0, odist * kernelCount * sizeof(cufftComplex));

        // timerMap["memset"].push_back(timer.elapsed());

        // timer.restart();
        // timerMap["memcpy"].push_back(timer.elapsed());

        // padded signal
        cudaMemcpy(d_signal, Signal, dataSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernels, Kernels, kernelSize * kernelCount * sizeof(float), cudaMemcpyHostToDevice);
        dim3 threadBlock(16, 16, 4);
        dim3 dataBlockGrid(iDivUp(fftCols, threadBlock.x), iDivUp(fftRows, threadBlock.y), 1);
        dim3 kernelBlockGrid(iDivUp(fftCols, threadBlock.x), iDivUp(fftRows, threadBlock.y), iDivUp(kernelCount, threadBlock.z));
        //dim3 kernelBlockGrid(iDivUp(kCols, threadBlock.x), iDivUp(kRows, threadBlock.y), iDivUp(kernelCount, threadBlock.z));
        padDataWithZeros<<<dataBlockGrid, threadBlock>>>(d_paddedSignal, d_signal, fftCols, fftRows, nCols, nRows, 1); 
        padDataWithZeros<<<kernelBlockGrid, threadBlock>>>(d_paddedKernels, d_kernels, fftCols, fftRows, kCols, kRows, kernelCount);

#if VERBOSE
        CudaCheckError();
        float  *h_paddedSignal = (float*)malloc(sizeof(float) * fftSize);  // padded signal
        float  *h_paddedKernels = (float*)malloc(sizeof(float) * fftSize * kernelCount);  // padded kernels

        cudaMemcpy(h_paddedSignal, d_paddedSignal, sizeof(float) * fftSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_paddedKernels, d_paddedKernels, sizeof(float) * fftSize * kernelCount, cudaMemcpyDeviceToHost);

        for(int i = 0;i<10;i++){
            for(int j = 0;j<10;j++)
            	printf("%f ", h_paddedSignal[i * fftCols + j]);
            printf("\n");
	}
        printf("\n");
        for(int i = 0;i<10;i++){
            for(int j = 0;j<10;j++)
            	printf("%f ", h_paddedKernels[i * fftCols + j]);
            printf("\n");
	}
        printf("\n");
#endif
        cufftPlan2d(&fwplan_signal, fftRows, fftCols, CUFFT_R2C);
        cufftPlanMany(&fwplan_kernels, 2, n, inembed, 1, idist, onembed, 1, odist, CUFFT_R2C, kernelCount);
        cufftPlanMany(&bwplan, 2, n, onembed, 1, odist, inembed, 1, idist, CUFFT_C2R, kernelCount);
        // timer.restart();
        // timerMap["create_plans"].push_back(timer.elapsed());

	// warm up
	{
            // timer.restart();
            cufftExecR2C(fwplan_signal, d_paddedSignal, d_signal_fft);
            cufftExecR2C(fwplan_kernels, d_paddedKernels, d_kernels_fft);
            // timerMap["forward"].push_back(timer.elapsed());

            // timer.restart();

            thrust::device_ptr<cufftComplex> thr_signal_fft(d_signal_fft);
            thrust::device_ptr<cufftComplex> thr_kernel_fft(d_kernels_fft);
            //thrust::transform(thr_signal_fft, thr_signal_fft + kernelCount * cfftSize, thr_kernel_fft, thr_kernel_fft, complexMultiplier(1.0 / fftSize));
            for(int i = 0 ;i < kernelCount; i++){
                thrust::transform(thr_signal_fft, thr_signal_fft + cfftSize, thr_kernel_fft + i * cfftSize, thr_kernel_fft + i * cfftSize,
                                  complexMultiplier(1.0 / fftSize));
                // timerMap["multiply"].push_back(timer.elapsed());
            }

#if VERBOSE
            size_t freeMem, totalMem;
            cudaMemGetInfo(&freeMem, &totalMem);
            std::cout << "Free: " << freeMem << ", Total: " << totalMem << std::endl;
#endif

            // timer.restart();
            cufftExecC2R(bwplan, d_kernels_fft, d_results);
            //cufftExecZ2Z(*bwplan, d_kernels_fft, d_kernels, CUFFT_INVERSE);
            // timerMap["backward"].push_back(timer.elapsed());
	}

        cudaDeviceSynchronize();
	int fft_time = 0;
	int dot_product_time = 0;
	int ifft_time = 0;

	    int duplicate = 8;
		cufftComplex* d_double_signal_fft;
        cudaMalloc((void**)&d_double_signal_fft, cfftSize * duplicate * sizeof(cufftComplex));
        int num_repeats = NUM_ITERATIONS;
        for (int i = 0; i < num_repeats; ++i) 
	{
            // timer.restart();
            auto start = std::chrono::steady_clock::now();
            cufftExecR2C(fwplan_signal, d_paddedSignal, d_signal_fft);
            cufftExecR2C(fwplan_kernels, d_paddedKernels, d_kernels_fft);
            cudaDeviceSynchronize();
            auto end = std::chrono::steady_clock::now();
            fft_time += static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() );
            // timerMap["forward"].push_back(timer.elapsed());

            // timer.restart();

            start = std::chrono::steady_clock::now();

			//// use thrust
			//for(int c = 0; c < duplicate;c++)
            //    cudaMemcpy(d_double_signal_fft + c * cfftSize, d_signal_fft, sizeof(cufftComplex) * cfftSize, cudaMemcpyDeviceToDevice);
            ////thrust::device_ptr<cufftComplex> thr_signal_fft(d_signal_fft);
            //thrust::device_ptr<cufftComplex> thr_signal_fft(d_double_signal_fft);
            //thrust::device_ptr<cufftComplex> thr_kernel_fft(d_kernels_fft);
            ////thrust::transform(thr_signal_fft, thr_signal_fft + kernelCount * cfftSize, thr_kernel_fft, thr_kernel_fft, complexMultiplier(1.0 / fftSize));
            //for(int i = 0 ;i < kernelCount / duplicate; i++){
            //    thrust::transform(thr_signal_fft, thr_signal_fft + cfftSize * duplicate, thr_kernel_fft + i * duplicate * cfftSize, thr_kernel_fft + i * duplicate * cfftSize,
            //                      complexMultiplier(1.0 / fftSize));
            //    // timerMap["multiply"].push_back(timer.elapsed());
            //}

            dim3 dotBlock(32, 32);
            dim3 dotGrid(fftCols / 32, iDivUp((fftRows / 2 + 1), 32), 16);
			dotProduct<<<dotGrid, dotBlock>>>(d_signal_fft, fftCols, fftRows / 2 + 1, 1.0 / fftSize, kernelCount/16, d_kernels_fft);

            cudaDeviceSynchronize();
            end = std::chrono::steady_clock::now();
            dot_product_time += static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() );

#if VERBOSE
            size_t freeMem, totalMem;
            cudaMemGetInfo(&freeMem, &totalMem);
            std::cout << "Free: " << freeMem << ", Total: " << totalMem << std::endl;
#endif

            // timer.restart();
            start = std::chrono::steady_clock::now();
            cufftExecC2R(bwplan, d_kernels_fft, d_results);
            //cufftExecZ2Z(*bwplan, d_kernels_fft, d_kernels, CUFFT_INVERSE);
            // timerMap["backward"].push_back(timer.elapsed());
            cudaDeviceSynchronize();
            end = std::chrono::steady_clock::now();
            ifft_time += static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() );
	}

        int avg_fft_time = fft_time / num_repeats;
        int avg_dot_product_time = dot_product_time / num_repeats;
        int avg_ifft_time = ifft_time / num_repeats;
	int avg_total_time = avg_fft_time + avg_dot_product_time + avg_ifft_time;
	printf("direct_fft total time:%d\n", avg_total_time);

        // timer.restart();
        cudaMemcpy(results, d_results, kernelCount * fftSize * sizeof(float), cudaMemcpyDeviceToHost);
       
        // timerMap["copy_back"].push_back(timer.elapsed());
        // timerMap["total_time"].push_back(globalTimer.elapsed());

        // free GPU memory
        cudaFree(d_signal);
        cudaFree(d_paddedSignal);
        cudaFree(d_kernels);
        cudaFree(d_paddedKernels);
        cudaFree(d_results);
        cudaFree(d_signal_fft);
        cudaFree(d_kernels_fft);

    }

    struct saxpy_functor
    {
        const float a;
    
        saxpy_functor(float _a) : a(_a) {}
    
        __host__ __device__
            float operator()(const float& x, const float& y) const { 
                return a * x + y;
            }
    };
    
    void saxpy_fast(float A, thrust::device_ptr<float>& X, thrust::device_ptr<float>& Y, int matrixSize)
    {
        // Y <- A * X + Y
        thrust::transform(X, X + matrixSize, Y, Y, saxpy_functor(A));
    }



    void convolve_turbo_fft_v1(float * Signal, float * Kernels, int kernelCount, 
                             int nRows, int nCols, int kRows, int kCols, float * results)
    {

        cufftHandle fwplan_signal, fwplan_kernels, bwplan; 
        // timer timer, globalTimer;
        // globalTimer.restart();

        int fftRows = nRows + kRows - 1;
        int fftCols = nCols + kCols - 1;

        int dataSize = nRows * nCols;
        int kernelSize = kRows * kCols;
        int fftSize = fftRows * fftCols;
        int cfftSize = fftRows * (fftCols / 2 + 1);

        // initilize basic one kernels and basic signal maps
        float *h_one_kernels = NULL;
        float *h_basic_signals = NULL;
        h_one_kernels = (float*)malloc(sizeof(float) * fftSize * kernelSize);
        h_basic_signals = (float*)malloc(sizeof(float) * fftSize * kernelSize);
        memset(h_one_kernels, 0, sizeof(float) * fftSize * kernelSize);
        memset(h_basic_signals, 0, sizeof(float) * fftSize * kernelSize);

        for(int i = 0; i < kRows; i++)
            for(int j = 0; j < kCols; j++){
                int kerIdx = i * kCols + j;
                int oneIdx = i * fftCols + j;
                h_one_kernels[kerIdx * fftSize + oneIdx] = 1.0;
        }
#if DEBUG
        for(int i = 0; i < kernelSize; i++){
            for(int j = 0; j < fftRows; j++){
        	for(int k = 0; k < fftCols; k++)
                    printf("%f ", h_one_kernels[i * fftSize + j * fftCols + k]);
                printf("\n");
            }
            printf("\n");
        }
#endif
        float *d_one_kernels = NULL;
        cudaMalloc((void**)&d_one_kernels, sizeof(float) * fftSize * kernelSize);
        cudaMemcpy(d_one_kernels, h_one_kernels, sizeof(float) * fftSize * kernelSize, cudaMemcpyHostToDevice);

        // device memory pointer
        float *d_signal = NULL;   // device storage of input signal
        float *d_paddedSignal = NULL;   // device storage of input signal
        float *d_basic_signals = NULL;   // device storage of input signal
        float *d_results = NULL;
        
        cudaMalloc((void**)&d_signal, sizeof(float) * dataSize);
        cudaMalloc((void**)&d_paddedSignal, sizeof(float) * fftSize);
        //cudaMalloc((void**)&d_basic_signals, sizeof(float) * fftSize * kernelSize);
        cudaMalloc((void**)&d_results, sizeof(float) * fftSize * kernelCount);

        // timer.restart();
        cudaMemset(d_signal, 0, dataSize * sizeof(float));
        cudaMemset(d_paddedSignal, 0, fftSize * sizeof(float));
        //cudaMemset(d_basic_signals, 0, fftSize * kernelSize * sizeof(float));
        cudaMemset(d_results, 0, fftSize * kernelCount * sizeof(float));
        // timerMap["memset"].push_back(timer.elapsed());

        //for(int i = 0;i<10;i++){
        //    printf("%f\n", paddedSignal[i]);
        //    printf("%f\n", paddedKernels[i]);
        //}
        // timer.restart();
        cudaMemcpy(d_signal, Signal, dataSize * sizeof(float), cudaMemcpyHostToDevice);
        // padded signal
        dim3 threadBlock(16, 16, 4);
        dim3 BlockGrid(iDivUp(fftCols, threadBlock.x), iDivUp(fftRows, threadBlock.y), 1);
        padDataWithZeros<<<BlockGrid, threadBlock>>>(d_paddedSignal, d_signal, fftCols, fftRows, nCols, nRows, 1);      

#if DEBUG
        float *h_paddedSignal = (float*)malloc(sizeof(float) * fftSize);
        cudaMemcpy(h_paddedSignal, d_paddedSignal, fftSize * sizeof(float), cudaMemcpyDeviceToHost);
        for(int i = 0;i<fftRows;i++){
            for(int j = 0;j<fftCols;j++)
            	printf("%f ", h_paddedSignal[i * fftCols + j]);
            printf("\n");
	}
        printf("\n");
        // timerMap["memcpy"].push_back(timer.elapsed());
#endif

        cufftComplex *d_signal_fft;
        cufftComplex *d_one_kernels_fft;

        cudaMalloc((void**)&d_signal_fft, cfftSize * sizeof(cufftComplex));
        cudaMalloc((void**)&d_one_kernels_fft, cfftSize * kernelSize * sizeof(cufftComplex));
        //cudaMemset(d_signal_fft, 0, cfftSize * sizeof(cufftComplex));
        //cudaMemset(d_one_kernels_fft, 0, cfftSize * kernelSize * sizeof(cufftComplex));

        // for cufftPlanMany parameters
        int idist = fftRows * fftCols;
        int odist = fftRows * (fftCols / 2 + 1);
        int inembed[] = {fftRows, fftCols};
        int onembed[] = {fftRows, fftCols / 2 + 1};
        int n[2] = {fftRows, fftCols};

        cufftPlan2d(&fwplan_signal, fftRows, fftCols, CUFFT_R2C);
        cufftPlanMany(&fwplan_kernels, 2, n, inembed, 1, idist, onembed, 1, odist, CUFFT_R2C, kernelSize);
        cufftPlanMany(&bwplan, 2, n, onembed, 1, odist, inembed, 1, idist, CUFFT_C2R, kernelSize);
        //cufftPlan1d(&fwplan_signal, nCols, CUFFT_R2C, 1);
        //cufftPlan1d(&fwplan_kernels, nCols, CUFFT_R2C, kernelCount);
        //cufftPlan1d(&bwplan, nCols, CUFFT_C2R, kernelCount);

        // timer.restart();
        //create_plan(fwplan_signal, real_size, CUFFT_Z2Z, 1);
        //create_plan(fwplan_kernels, real_size, CUFFT_Z2Z, kernelCount);
        //create_plan(bwplan, real_size, CUFFT_Z2Z, kernelCount);
        // timerMap["create_plans"].push_back(timer.elapsed());

        // timer.restart();
        cufftExecR2C(fwplan_signal, d_paddedSignal, d_signal_fft);
        cufftExecR2C(fwplan_kernels, d_one_kernels, d_one_kernels_fft);
        // timerMap["forward"].push_back(timer.elapsed());

#if DEBUG
        cufftComplex *h_one_kernels_fft = (cufftComplex*)malloc(sizeof(cufftComplex) * kernelSize * cfftSize);
        cudaMemcpy(h_one_kernels_fft, d_one_kernels_fft, sizeof(cufftComplex) * kernelSize * cfftSize, cudaMemcpyDeviceToHost);
        for(int k = 0 ; k < kernelSize; k++){
            for (int i = 0 ; i < cfftSize; i++)
                printf("%.2f+%.2fj ", h_one_kernels_fft[k * cfftSize + i].x, h_one_kernels_fft[k * cfftSize + i].y);
            printf("\n");
        }
        printf("\n");
#endif

        // timer.restart();
        thrust::device_ptr<cufftComplex> thr_signal_fft(d_signal_fft);
        for(int i = 0 ;i < kernelSize; i++){
            thrust::device_ptr<cufftComplex> thr_one_kernel_fft(d_one_kernels_fft + i * odist);
            thrust::transform(thr_signal_fft, thr_signal_fft + odist, thr_one_kernel_fft, thr_one_kernel_fft,
                              complexMultiplier(1.0 / idist));
            // timerMap["multiply"].push_back(timer.elapsed());
        }

        // timer.restart();
        //cufftExecC2R(bwplan, d_one_kernels_fft, d_basic_signals);
        cufftExecC2R(bwplan, d_one_kernels_fft, d_one_kernels);
        d_basic_signals = d_one_kernels;

#if DEBUG
        cudaMemcpy(h_basic_signals, d_basic_signals, kernelSize * fftSize * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0 ; i < kernelSize;i++){
	    for(int r = 0 ; r < fftRows; r++){
		for(int c = 0; c< fftCols;c++)
		    printf("%f ", h_basic_signals[i * fftSize + r * fftCols + c]);
                printf("\n");
	    }
	    printf("\n");
        }
	printf("\n");
#endif
        //cufftExecZ2Z(*bwplan, d_kernels_fft, d_kernels, CUFFT_INVERSE);
        // timerMap["backward"].push_back(timer.elapsed());

        // timer.restart();
        // cudaMemcpy(h_basic_signals, d_one_kernels_fft, kernelSize * idist * sizeof(float), cudaMemcpyDeviceToHost);
       
        // timerMap["copy_back"].push_back(timer.elapsed());
        // timerMap["total_time"].push_back(globalTimer.elapsed());

        
        //for (int r = 0 ; r < fftRows; r++)
        //    for(int c = 0; c < fftCols; c++){
        //	for(int j = 0 ; j < kernelSize; j++){
        //            float a1 = results[r * fftCols + c];
        //            float a2 = h_basic_signals[j * fftSize + r * fftCols + c];
	//    	    results[r * fftCols + c] += h_basic_signals[j * fftSize + r * fftCols + c];
        //            printf("%f+%f=%f ", a1, a2, results[r * fftCols + c]);
	//	}
	//    printf("\n");
        //}
        //for(int i = 0; i < kernelCount; i++){
        //    for(int j = 0 ; j < kernelSize; j++){
        //        for (int r = 0 ; r < fftRows; r++)
        //            for(int c = 0; c < fftCols; c++)        
	//		results[i * fftSize + r * fftCols + c] += Kernels[i * kernelSize + j] * h_basic_signals[j * fftSize + r * fftCols + c];
        //    }
        //}


        for(int i = 0; i < kernelCount; i++){
            thrust::device_ptr<float> thr_results(d_results + i * fftSize);
            for(int j = 0 ; j < kernelSize; j++){
                thrust::device_ptr<float> thr_basic_signals(d_basic_signals + j * fftSize);
		saxpy_fast(Kernels[i * kernelSize + j], thr_basic_signals, thr_results, fftSize);
            }
        }

        cudaMemcpy(results, d_results, kernelCount * fftSize * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_one_kernels);
        cudaFree(d_signal);
        cudaFree(d_paddedSignal);
        cudaFree(d_results);
        cudaFree(d_signal_fft);
        cudaFree(d_one_kernels_fft);

    }

    void convolve_turbo_fft_v2(float * Signal, float * Kernels, int kernelCount, 
                             int nRows, int nCols, int kRows, int kCols, float * results)
    {

        cufftHandle fwplan_signal, fwplan_kernels, bwplan; 
        // timer timer, globalTimer;
        // globalTimer.restart();

        int fftRows = nRows + kRows - 1;
        int fftCols = nCols + kCols - 1;
        int dataSize = nRows * nCols;
        int kernelSize = kRows * kCols;
        int fftSize = fftRows * fftCols;
        int cfftSize = fftRows * fftCols;
        int group_num = kernelCount / 2;

	// parameter for cufftPlanMany
        int idist = fftRows * fftCols;
        int odist = fftRows * fftCols;
        int inembed[] = {fftRows, fftCols};
        int onembed[] = {fftRows, fftCols};
        int n[2] = {fftRows, fftCols};

#if DEBUG
        for(int i = 0;i<nRows;i++){
            for(int j = 0;j<nCols;j++)
                printf("%f ", Signal[i * nCols + j]);
            printf("\n");
        }
        printf("\n");
        for(int i = 0;i<kRows;i++){
            for(int j = 0;j<kCols;j++)
                printf("%f ", Kernels[i * kCols + j]);
            printf("\n");
	}
        printf("\n");
#endif

        // device memory pointer
        float *d_signal = NULL;   // device storage of input signal
        float *d_kernels = NULL;  // device storage of kernels and final results
        cufftComplex *d_paddedSignal = NULL;
        cufftComplex *d_paddedKernels = NULL;  // device storage of kernels and final results
        float *d_results = NULL;
        cufftComplex *d_signal_fft;
        cufftComplex *d_kernels_fft;
        cufftComplex *d_w_results;
        
        cudaMalloc((void**)&d_signal, sizeof(float) * dataSize);
        cudaMalloc((void**)&d_kernels, sizeof(float) * kernelSize * kernelCount);
        cudaMalloc((void**)&d_paddedSignal, sizeof(cufftComplex) * fftSize);
        cudaMalloc((void**)&d_paddedKernels, sizeof(cufftComplex) * fftSize * group_num);
        cudaMalloc((void**)&d_results, sizeof(float) * fftSize * kernelCount);
        cudaMalloc((void**)&d_signal_fft, cfftSize * sizeof(cufftComplex));
        cudaMalloc((void**)&d_kernels_fft, cfftSize * group_num * sizeof(cufftComplex));
        cudaMalloc((void**)&d_w_results, cfftSize * group_num * sizeof(cufftComplex));

        // timer.restart();
        cudaMemset(d_signal, 0, dataSize * sizeof(float));
        cudaMemset(d_kernels, 0, kernelSize * kernelCount * sizeof(float));
        cudaMemset(d_paddedSignal, 0, fftSize * sizeof(cufftComplex));
        cudaMemset(d_paddedKernels, 0, fftSize * group_num * sizeof(cufftComplex));
        cudaMemset(d_results, 0, fftSize * kernelCount * sizeof(float));
        // timerMap["memset"].push_back(timer.elapsed());

        // timer.restart();
        cudaMemcpy(d_signal, Signal, dataSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernels, Kernels, kernelSize * kernelCount * sizeof(float), cudaMemcpyHostToDevice);
        //cudaMemcpy(Kernels, d_kernels, kernelSize * kernelCount * sizeof(float), cudaMemcpyDeviceToHost);
        //for(int i = 0;i<kernelSize;i++)
        //    printf("%f\n", Kernels[i]);
        //printf("\n");
        // timerMap["memcpy"].push_back(timer.elapsed());

        // padded signal
        dim3 threadBlock(16, 16, 4);
        dim3 dataBlockGrid(iDivUp(fftCols, threadBlock.x), iDivUp(fftRows, threadBlock.y), 1);
        dim3 kernelBlockGrid(iDivUp(fftCols, threadBlock.x), iDivUp(fftRows, threadBlock.y), iDivUp(group_num, threadBlock.z));
        //dim3 kernelBlockGrid(iDivUp(kCols, threadBlock.x), iDivUp(kRows, threadBlock.y), iDivUp(group_num, threadBlock.z));
        //dim3 kernelBlockGrid(1, 1, iDivUp(group_num, threadBlock.z));
	    int padding_time = 0;
        auto start_pad = std::chrono::steady_clock::now();

        padDataWithZerosComplex<<<dataBlockGrid, threadBlock>>>(d_paddedSignal, d_signal, fftCols, fftRows, nCols, nRows, 1); 
        padKernelWithZerosComplex<<<kernelBlockGrid, threadBlock>>>(d_paddedKernels, d_kernels, fftCols, fftRows, kCols, kRows, group_num);
        cudaDeviceSynchronize();

        auto end_pad = std::chrono::steady_clock::now();
        padding_time += static_cast<int>(std::chrono::duration<double, std::micro>(end_pad - start_pad).count() );
	//printf("padding time for %d group(s): %d.\n", group_num, padding_time);

#if DEBUG
        CudaCheckError();
        cufftComplex *h_paddedSignal = (cufftComplex*)malloc(sizeof(cufftComplex) * fftSize);
        cufftComplex *h_paddedKernels = (cufftComplex*)malloc(sizeof(cufftComplex) * fftSize * group_num);

        cudaMemcpy(h_paddedSignal, d_paddedSignal, sizeof(cufftComplex) * fftSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_paddedKernels, d_paddedKernels, sizeof(cufftComplex) * fftSize * group_num, cudaMemcpyDeviceToHost);

        for(int i = 0;i<fftRows;i++){
            for(int j = 0;j<fftCols;j++)
            	printf("%f+%fi ", h_paddedSignal[i * fftCols + j].x, h_paddedSignal[i * fftCols + j].y);
            printf("\n");
	}
        printf("\n");
        for(int i = 0;i<fftRows;i++){
            for(int j = 0;j<fftCols;j++)
            	printf("%f+%fi ", h_paddedKernels[i * fftCols + j].x, h_paddedKernels[i * fftCols + j].y);
            printf("\n");
	}
        printf("\n");
		getchar();
#endif
        //cudaMemset(d_signal_fft, 0, odist * sizeof(cufftComplex));
        //cudaMemset(d_kernels_fft, 0, odist * kernelCount * sizeof(cufftComplex));

        cufftPlan2d(&fwplan_signal, fftRows, fftCols, CUFFT_C2C);
        cufftPlanMany(&fwplan_kernels, 2, n, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, group_num);
        cufftPlanMany(&bwplan, 2, n, onembed, 1, odist, inembed, 1, idist, CUFFT_C2C, group_num);
        //cufftPlan1d(&fwplan_signal, nCols, CUFFT_R2C, 1);
        //cufftPlan1d(&fwplan_kernels, nCols, CUFFT_R2C, kernelCount);
        //cufftPlan1d(&bwplan, nCols, CUFFT_C2R, kernelCount);

        // timer.restart();
        //create_plan(fwplan_signal, real_size, CUFFT_Z2Z, 1);
        //create_plan(fwplan_kernels, real_size, CUFFT_Z2Z, kernelCount);
        //create_plan(bwplan, real_size, CUFFT_Z2Z, kernelCount);
        // timerMap["create_plans"].push_back(timer.elapsed());

        // timer.restart();

	// warm up
	{

            cufftExecC2C(fwplan_signal, d_paddedSignal, d_signal_fft, CUFFT_FORWARD);
            cufftExecC2C(fwplan_kernels, d_paddedKernels, d_kernels_fft, CUFFT_FORWARD);
            // timerMap["forward"].push_back(timer.elapsed());

            // timer.restart();

            dim3 dotBlock(32, 32);
            dim3 dotGrid(fftCols / 32, fftCols / 32, 8);
	    dotProduct<<<dotGrid, dotBlock>>>(d_signal_fft, fftCols, fftRows, 1.0 / fftSize, group_num/8, d_kernels_fft);
            cudaDeviceSynchronize();

            // thrust::device_ptr<cufftComplex> thr_signal_fft(d_signal_fft);
            // thrust::device_ptr<cufftComplex> thr_kernel_fft(d_kernels_fft);
            // //thrust::transform(thr_signal_fft, thr_signal_fft + group_num * cfftSize, thr_kernel_fft, thr_kernel_fft, complexMultiplier(1.0 / fftSize));
            // for(int i = 0 ;i < group_num; i++){
            //     thrust::transform(thr_signal_fft, thr_signal_fft + cfftSize, thr_kernel_fft + i * cfftSize, thr_kernel_fft + i * cfftSize,
            //                       complexMultiplier(1.0 / fftSize));
            //     // timerMap["multiply"].push_back(timer.elapsed());
            // }

#if VERBOSE
            size_t freeMem, totalMem;
            cudaMemGetInfo(&freeMem, &totalMem);
            std::cout << "Free: " << freeMem << ", Total: " << totalMem << std::endl;
#endif

            //cudaMalloc((void**)&d_conj_w_results, cfftSize * group_num * sizeof(cufftComplex));
            // timer.restart();
            cufftExecC2C(bwplan, d_kernels_fft, d_w_results, CUFFT_INVERSE);
	}


        cudaDeviceSynchronize();

	// timing record
	int fft_time = 0;
	int dot_product_time = 0;
	int ifft_time = 0;

        int num_repeats = NUM_ITERATIONS;
		//cufftComplex* d_double_signal_fft;
		//int duplicate = 16;
        //cudaMalloc((void**)&d_double_signal_fft, fftSize * duplicate * sizeof(cufftComplex));
        for (int i = 0; i < num_repeats; ++i) 
	{

            auto start = std::chrono::steady_clock::now();
            cufftExecC2C(fwplan_signal, d_paddedSignal, d_signal_fft, CUFFT_FORWARD);
            cufftExecC2C(fwplan_kernels, d_paddedKernels, d_kernels_fft, CUFFT_FORWARD);
            cudaDeviceSynchronize();
            auto end = std::chrono::steady_clock::now();
            fft_time += static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() );

            // timerMap["forward"].push_back(timer.elapsed());

            // timer.restart();

	    start = std::chrono::steady_clock::now();

		    //// thrust implementation
			//for(int c = 0; c < duplicate;c++)
            //    cudaMemcpy(d_double_signal_fft + c * fftSize, d_signal_fft, sizeof(cufftComplex) * fftSize, cudaMemcpyDeviceToDevice);
            //thrust::device_ptr<cufftComplex> thr_signal_fft(d_double_signal_fft);
            //thrust::device_ptr<cufftComplex> thr_kernel_fft(d_kernels_fft);
            ////thrust::transform(thr_signal_fft, thr_signal_fft + group_num * cfftSize, thr_kernel_fft, thr_kernel_fft, complexMultiplier(1.0 / fftSize));
            //for(int i = 0 ;i < group_num / duplicate; i++){
            //    thrust::transform(thr_signal_fft, thr_signal_fft + duplicate * fftSize, thr_kernel_fft + i * duplicate * fftSize, thr_kernel_fft + i * duplicate * fftSize,
            //                      complexMultiplier(1.0 / fftSize));
            //    // timerMap["multiply"].push_back(timer.elapsed());
            //}
            dim3 dotBlock(32, 32);
            dim3 dotGrid(fftCols / 32, fftCols / 32, 8);
			dotProduct<<<dotGrid, dotBlock>>>(d_signal_fft, fftCols, fftRows, 1.0 / fftSize, group_num/8, d_kernels_fft);
            cudaDeviceSynchronize();
	    end = std::chrono::steady_clock::now();
            dot_product_time += static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() );

#if VERBOSE
            size_t freeMem, totalMem;
            cudaMemGetInfo(&freeMem, &totalMem);
            std::cout << "Free: " << freeMem << ", Total: " << totalMem << std::endl;
#endif

            //cudaMalloc((void**)&d_conj_w_results, cfftSize * group_num * sizeof(cufftComplex));
            // timer.restart();
	    start = std::chrono::steady_clock::now();
            cufftExecC2C(bwplan, d_kernels_fft, d_w_results, CUFFT_INVERSE);
            cudaDeviceSynchronize();
	    end = std::chrono::steady_clock::now();
            ifft_time += static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() );

	}

        int avg_fft_time = fft_time / num_repeats;
        int avg_dot_product_time = dot_product_time / num_repeats;
        int avg_ifft_time = ifft_time / num_repeats;
	int avg_total_time = avg_fft_time + avg_dot_product_time + avg_ifft_time;
	printf("turbo_fft_gpu_v2 total time:%d\n", avg_total_time);

        // split complex results
        SplitComplex<<<kernelBlockGrid, threadBlock>>>(d_results, d_w_results, fftCols, fftRows, group_num);
        //ConjComplex<<<kernelBlockGrid, threadBlock>>>(d_conj_w_results, d_w_results, fftCols, fftRows, group_num);
        //cufftExecZ2Z(*bwplan, d_kernels_fft, d_kernels, CUFFT_INVERSE);
        // timerMap["backward"].push_back(timer.elapsed());

        // timer.restart();
        cudaMemcpy(results, d_results, kernelCount * fftSize * sizeof(float), cudaMemcpyDeviceToHost);
       
        // timerMap["copy_back"].push_back(timer.elapsed());
        // timerMap["total_time"].push_back(globalTimer.elapsed());

        // free GPU memory
        cudaFree(d_signal);
        cudaFree(d_kernels);
        cudaFree(d_paddedSignal);
        cudaFree(d_paddedKernels);
        cudaFree(d_results);
        cudaFree(d_signal_fft);
        cudaFree(d_kernels_fft);
        cudaFree(d_w_results);

    }

    // data and qdVectors are in GPUs
    void qdVector_construct(const cufftComplex* d_data, int group_num, int N, int kCols, cufftComplex* d_qdVectors){

	int fftSize = N * N;

	bool use_gpu = 1;
	bool sparse = 1;
	if (use_gpu){
	    if(sparse){
	        // gpu sparse version of qdVector construction
                dim3 threadBlock(32, 1);
                dim3 BlockGrid(group_num, iDivUp(non_empty_qdV, 32));
	        //printf("group num: %d, qdV num: %d.\n", group_num, non_empty_qdV);
	        //printf("%d, %d, %d, %d.\n", threadBlock.x, threadBlock.y, BlockGrid.x, BlockGrid.y);
                qdVectorConstruct_sparse<<<BlockGrid, threadBlock>>>(d_data, d_sparse_input_xs, d_sparse_input_ys, d_begin_arr, d_end_arr, d_qdV_arr, N, kCols, group_num, non_empty_qdV, d_qdVectors); 
                cudaDeviceSynchronize();

/*
	        cufftComplex *h_qdVectors = (cufftComplex*)malloc(sizeof(cufftComplex) * fftSize * 3 / 2 * group_num);
	        cudaMemcpy(h_qdVectors, d_qdVectors, sizeof(cufftComplex) * fftSize * 3 / 2 * group_num, cudaMemcpyDeviceToHost);
                for(int i = 0;i< N * 3 / 2 ;i++){
	            printf("(%d %d): ", p[i], s[i]);
                    for(int j = 0;j<N;j++)
                    	printf("%f+%fi ", h_qdVectors[i * N + j].x, h_qdVectors[i * N + j].y);
                    printf("\n");
	            getchar();
	        }
                printf("\n");
*/

	    }
	    else{
	        // gpu version of qdVector construction
	        int qdV_num = N * 3 / 2;
                dim3 threadBlock(32, N / 32);
                dim3 BlockGrid(group_num, qdV_num);
	        printf("group num: %d, qdV num: %d.\n", group_num, qdV_num);
	        printf("%d, %d, %d, %d.\n", threadBlock.x, threadBlock.y, BlockGrid.x, BlockGrid.y);
                qdVectorConstruct<<<BlockGrid, threadBlock>>>(d_data, d_input_xs, d_input_ys, N, group_num, qdV_num, d_qdVectors); 
                cudaDeviceSynchronize();


	        cufftComplex *h_qdVectors = (cufftComplex*)malloc(sizeof(cufftComplex) * fftSize * 3 / 2 * group_num);
	        cudaMemcpy(h_qdVectors, d_qdVectors, sizeof(cufftComplex) * fftSize * 3 / 2 * group_num, cudaMemcpyDeviceToHost);
                for(int i = 0;i< N * 3 / 2 ;i++){
	            printf("(%d %d): ", p[i], s[i]);
                    for(int j = 0;j<N;j++)
                    	printf("%f+%fi ", h_qdVectors[i * N + j].x, h_qdVectors[i * N + j].y);
                    printf("\n");
	            getchar();
	        }
                printf("\n");

	    }
	}
	else
	// cpu version of qdVector construction
	{
            cufftComplex *h_paddedKernels = (cufftComplex*)malloc(sizeof(cufftComplex) * fftSize * group_num);  // padded kernels
            cudaMemcpy(h_paddedKernels, d_data, sizeof(cufftComplex) * fftSize * group_num, cudaMemcpyDeviceToHost);
	    cufftComplex *h_qdVectors = (cufftComplex*)malloc(sizeof(cufftComplex) * fftSize * 3 / 2 * group_num);
            memset(h_qdVectors, 0, sizeof(cufftComplex) * fftSize * 3 / 2 * group_num);

	    if (sparse){

	        // construct 1d vectors for 1d-fft
	        for(int k = 0; k < group_num;k++){
	            int kOffset = k * fftSize * 3 / 2;
	            for (int q = 0; q < non_empty_qdV; q++){
	                for (int s = begin_arr[q] ; s < end_arr[q];s++){
	                    int idx_x = sparse_input_xs[s];
	                    int idx_y = sparse_input_ys[s];
	                    h_qdVectors[kOffset + qdV_arr[q]].x += h_paddedKernels[k * fftSize + idx_x * N + idx_y].x;
	                    h_qdVectors[kOffset + qdV_arr[q]].y += h_paddedKernels[k * fftSize + idx_x * N + idx_y].y;
		        }
		    }
		}
	    
	    }
	    else
	    {
#if DEBUG
	        printf("group_num: %d.\n", group_num);
                for(int i = 0;i<N;i++){
                    for(int j = 0;j<N;j++)
                    	printf("%f+%fi ", h_paddedKernels[i * N + j].x, h_paddedKernels[i * N + j].y);
                    printf("\n");
	        }
                printf("\n");
#endif

	        // construct 1d vectors for 1d-fft
	        for(int k = 0; k < group_num;k++){
	            int kOffset = k * fftSize * 3 / 2;
                    for(int i = 0; i < N + N/2;i++)
	                for(int t = 0; t < N; t++)
	                    for(int j = 0; j < N;j++){
	            	        int offset = i * fftSize + t * N + j;
	            	        int idx_x = input_xs[offset];
	            	        int idx_y = input_ys[offset];
	                        h_qdVectors[kOffset + i * N + t].x += h_paddedKernels[k * fftSize + idx_x * N + idx_y].x;
	                        h_qdVectors[kOffset + i * N + t].y += h_paddedKernels[k * fftSize + idx_x * N + idx_y].y;
	            	}
	        }

	    }

	    cudaMemcpy(d_qdVectors, h_qdVectors, sizeof(cufftComplex) * fftSize * 3 / 2 * group_num, cudaMemcpyHostToDevice);

#if DEBUG
        CudaCheckError();
        for(int i = 0;i<N * 3 / 2 ;i++){
	    printf("(%d %d):", p[i], s[i]);
            for(int j = 0;j<N;j++)
            	printf("%f+%fi ", h_qdVectors[i * N + j].x, h_qdVectors[i * N + j].y);
            printf("\n");
	    getchar();
	}
        printf("\n");
        printf("finish vector construction.\n");
#endif
	}


    }


    // fast fft with QD algorithm, the input signals and output results are on GPUs.
    void fast_fft(cufftComplex * d_kernels, int group_num, int fftCols, int fftRows, int kCols, int kRows, cufftComplex * d_results, bool forward = true)
    {

	int fftSize = fftCols * fftRows;

	// initialize space for qdVectors and ifft_results on GPUs
	if(d_qdVectors == NULL){

#if DEBUG
	    printf("Allocate gpu space of qdVectors...\n");
#endif
            cudaMalloc((void**)&d_qdVectors, sizeof(cufftComplex) * fftSize * 3 / 2 * group_num);
            cudaMalloc((void**)&d_w_results, sizeof(cufftComplex) * fftSize * 3 / 2 * group_num);
	}
	if (p == NULL){
#if DEBUG
	    printf("Construct index array for qd-fft for the first time...\n");
#endif
	    // initilize p and s
	    p = new int[fftCols + fftCols / 2];
	    s = new int[fftCols + fftCols / 2];
	    for(int i = 0 ; i < fftCols; i ++){
	        p[i] = 1;
	        s[i] = i;
	    }
	    for(int i = fftCols; i < fftCols + fftCols / 2;i++){
	        p[i] = (i - fftCols) * 2;
	        s[i] = 1;
	    }

#if DEBUG
            printf("finish initilization of (p, s).\n");
	    for(int i = 0 ; i < fftCols + fftCols / 2;i++)
	        printf("%d: p:%d, s:%d.\n", i, p[i], s[i]);
#endif

	    // initilize indices of get data, 1.5 * N * 2N^2, indices of out data, 
	    int N = fftCols;
	    input_xs = new int[N * 3 / 2 * (N * N)];
	    input_ys = new int[N * 3 / 2 * (N * N)];
	    output_xs = new int[N * 3 / 2 * N];
	    output_ys = new int[N * 3 / 2 * N];

            for(int i = 0; i < fftCols + fftCols/2;i++){
	        for(int t = 0 ; t < fftCols;t++){
	    	output_xs[i * fftCols + t] = (t * p[i]) % fftCols;
	    	output_ys[i * fftCols + t] = (t * s[i]) % fftCols;
	        }
	    }

	    for(int i = 0; i < fftCols + fftCols/2;i++)
	        for(int t = 0; t < fftCols; t++){
	    	int offset = i * fftSize + t * fftCols;
	            find_input_index(p[i], s[i], t, fftCols, input_xs + offset, input_ys + offset);
            }
#if DEBUG
            printf("finish indices search.\n");
	    for(int i = 0; i < fftCols + fftCols/2;i++)
	        for(int t = 0; t < fftCols; t++){
	    	if(p[i] != 18 || s[i] != 1)
	    		continue;
	    	printf("(%d, %d, %d):", p[i], s[i], t);
	    	for (int j = 0 ;j < fftCols;j++)
	    	    printf("[%d, %d]", input_xs[i * fftSize + t * fftCols + j], input_ys[i * fftSize + t * fftCols + j]);
	    	printf("\n");
	    	getchar();
	        }
#endif

	    if(sparse_input_xs == NULL){
	        non_zero = 0;
	        non_empty_qdV = 0;
	        for(int i = 0; i < fftCols + fftCols/2;i++)
	            for(int t = 0; t < fftCols; t++){
	                int tmp_non_zero = 0;
	                for(int j = 0; j < fftCols;j++)
	            	if(input_xs[i * fftSize + t * fftCols + j] < kCols && input_ys[i * fftSize + t * fftCols + j] < kCols)
	            	    tmp_non_zero++;

	                if(tmp_non_zero != 0){
	            	    non_empty_qdV++;
	            	    non_zero += tmp_non_zero;
	                }
	            }
	            
#if DEBUG
		printf("non_zero: %d, non_empty_qdV: %d.\n", non_zero, non_empty_qdV);
#endif
    	        sparse_input_xs = new int[non_zero];
    	        sparse_input_ys = new int[non_zero];
    	        begin_arr = new int[non_empty_qdV];
    	        end_arr = new int[non_empty_qdV];
    	        qdV_arr = new int[non_empty_qdV];

		int start_idx = 0;
		int end_idx = 0;
		int qdV_idx = 0;
	        for(int i = 0; i < fftCols + fftCols/2;i++)
	            for(int t = 0; t < fftCols; t++){
	                for(int j = 0; j < fftCols;j++)
	            	    if(input_xs[i * fftSize + t * fftCols + j] < kCols && input_ys[i * fftSize + t * fftCols + j] < kCols)
			    {
				sparse_input_xs[end_idx] = input_xs[i * fftSize + t * fftCols + j];
				sparse_input_ys[end_idx] = input_ys[i * fftSize + t * fftCols + j];
				end_idx++;
			    }

	                if(end_idx != start_idx){
	            	    begin_arr[qdV_idx] = start_idx;
	            	    end_arr[qdV_idx] = end_idx;
			    start_idx = end_idx;
			    qdV_arr[qdV_idx] = i * fftCols + t;
			    qdV_idx++;
	                }
	            }

		//// print out for check
	        //for (int q = 0; q < non_empty_qdV; q++){
	        //    printf("%d[%d]: ", q, qdV_arr[q]);
	        //    for (int s = begin_arr[q] ; s < end_arr[q];s++)
	        //        printf("(%d, %d) ", sparse_input_xs[s], sparse_input_ys[s]);
	        //    printf("\n");
		//    getchar();
	        //}
	    }
	    if(d_input_xs == NULL){
		// initialize GPU index array
                cudaMalloc((void**)&d_input_xs, sizeof(int) * N * 3 / 2 * fftSize);
                cudaMalloc((void**)&d_input_ys, sizeof(int) * N * 3 / 2 * fftSize);
                cudaMalloc((void**)&d_output_xs, sizeof(int) * N * 3 / 2 * N);
                cudaMalloc((void**)&d_output_ys, sizeof(int) * N * 3 / 2 * N);
		
	        cudaMemcpy(d_input_xs, input_xs, sizeof(int) * N * 3 / 2 * fftSize, cudaMemcpyHostToDevice);
	        cudaMemcpy(d_input_ys, input_ys, sizeof(int) * N * 3 / 2 * fftSize, cudaMemcpyHostToDevice);
	        cudaMemcpy(d_output_xs, output_xs, sizeof(int) * N * 3 / 2 * N, cudaMemcpyHostToDevice);
	        cudaMemcpy(d_output_ys, output_ys, sizeof(int) * N * 3 / 2 * N, cudaMemcpyHostToDevice);
	    }
	    if(d_sparse_input_xs == NULL){
		// initialize GPU index array
                cudaMalloc((void**)&d_sparse_input_xs, sizeof(int) * non_zero);
                cudaMalloc((void**)&d_sparse_input_ys, sizeof(int) * non_zero);
                cudaMalloc((void**)&d_begin_arr, sizeof(int) * non_empty_qdV);
                cudaMalloc((void**)&d_end_arr, sizeof(int) * non_empty_qdV);
                cudaMalloc((void**)&d_qdV_arr, sizeof(int) * non_empty_qdV);

	        cudaMemcpy(d_sparse_input_xs, sparse_input_xs, sizeof(int) * non_zero, cudaMemcpyHostToDevice);
	        cudaMemcpy(d_sparse_input_ys, sparse_input_ys, sizeof(int) * non_zero, cudaMemcpyHostToDevice);
		cudaMemcpy(d_begin_arr, begin_arr, sizeof(int) * non_empty_qdV, cudaMemcpyHostToDevice);
		cudaMemcpy(d_end_arr, end_arr, sizeof(int) * non_empty_qdV, cudaMemcpyHostToDevice);
		cudaMemcpy(d_qdV_arr, qdV_arr, sizeof(int) * non_empty_qdV, cudaMemcpyHostToDevice);
	    } 

	}

	int vec_con_time = 0;
        auto start = std::chrono::steady_clock::now();
        qdVector_construct(d_kernels, group_num, fftCols, kCols, d_qdVectors);
	//cudaMemcpy(d_qdVectors, h_qdVectors, sizeof(cufftComplex) * fftSize * 3 / 2 * group_num, cudaMemcpyHostToDevice);
        auto end = std::chrono::steady_clock::now();
        vec_con_time += static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() );

#if QDFFT_MEASURE
	printf("vec_construct:%d, ", vec_con_time);
#endif

	int fft_time = 0;
        start = std::chrono::steady_clock::now();
	if (forward)
            cufftExecC2C(plan1d, d_qdVectors, d_w_results, CUFFT_FORWARD);
	else
            cufftExecC2C(plan1d, d_qdVectors, d_w_results, CUFFT_INVERSE);

        cudaDeviceSynchronize();
        end = std::chrono::steady_clock::now();
        fft_time += static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() );

#if QDFFT_MEASURE
	printf("fft_time:%d, ", fft_time);
#endif

#if DEBUG
        printf("finish fft.\n");
#endif

	int use_cpu = 0;
	int re_arrange_time = 0;
        start = std::chrono::steady_clock::now();
	if (use_cpu)
	// cpu version of re-arrange data
	{
	    cufftComplex* h_results = (cufftComplex*)malloc(group_num * fftSize * sizeof(cufftComplex));
	    cufftComplex* h_w_results = (cufftComplex*)malloc(group_num * fftSize * 3 / 2 * sizeof(cufftComplex));
            cudaMemcpy(h_w_results, d_w_results, group_num * fftSize * 3 / 2 * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	    for(int k = 0; k < group_num;k++){
	        int kOffset = k * fftSize * 3 / 2;
                for(int i = 0; i < fftCols + fftCols/2;i++)
	            for(int t = 0; t < fftCols; t++){
	    	    int idx_x = output_xs[i * fftCols + t];
	    	    int idx_y = output_ys[i * fftCols + t];
	    	    h_results[k * fftSize + idx_x * fftCols + idx_y].x = h_w_results[kOffset + i * fftCols + t].x;
	    	    h_results[k * fftSize + idx_x * fftCols + idx_y].y = h_w_results[kOffset + i * fftCols + t].y;
	    	}
	    	
	    }

#if DEBUG
            CudaCheckError();
            for(int i = 0;i<fftRows;i++){
                for(int j = 0;j<fftCols;j++)
                	printf("%f+%fi ", h_results[i * fftCols + j].x, h_results[i * fftCols + j].y);
                printf("\n");
	    }
            printf("\n");
	    getchar();
#endif

	    cudaMemcpy(d_results, h_results, sizeof(cufftComplex) * fftSize * group_num, cudaMemcpyHostToDevice);
	}
	else
	{
	    int N = fftCols;

	    ////global memory
            //dim3 threadBlock(32, 1);
            //dim3 BlockGrid(group_num, iDivUp(N * N * 3 / 2, 32));
            //qdVectorReconstruct<<<BlockGrid, threadBlock>>>(d_w_results, d_output_xs, d_output_ys, N, group_num, d_results); 

	    //shared memory
		    int rows = 4;
            //dim3 top_half_threadBlock(N, 4);
            //dim3 top_half_BlockGrid(group_num, N/4);
            //qdVectorReconstruct_Shared<<<top_half_BlockGrid, top_half_threadBlock>>>(d_w_results, d_output_xs, d_output_ys, N, group_num, d_results, true); 
            dim3 top_half_threadBlock(N, N/4);
            dim3 top_half_BlockGrid(group_num, 1);
            qdVectorReconstruct_Small<<<top_half_BlockGrid, top_half_threadBlock>>>(d_w_results, d_output_xs, d_output_ys, N, group_num, d_results, true); 
            //dim3 threadBlock(N / 2, 1);
            //dim3 BlockGrid(group_num, 1);
            //qdVectorReconstruct_Shared<<<BlockGrid, threadBlock>>>(d_w_results, d_output_xs, d_output_ys, N, group_num, d_results, false); 
            cudaDeviceSynchronize();

	
	}
        end = std::chrono::steady_clock::now();
        re_arrange_time += static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() );

#if QDFFT_MEASURE
	printf("data_rearrange:%d.\n", re_arrange_time);
#endif

    }

    void convolve_turbo_fft_v3(float * Signal, float * Kernels, int kernelCount, 
                             int nRows, int nCols, int kRows, int kCols, float * results)
    {

        cufftHandle fwplan_signal, fwplan_kernels, bwplan; 
        // timer timer, globalTimer;
        // globalTimer.restart();

        int fftRows = nRows + kRows - 1;
        int fftCols = nCols + kCols - 1;
        int dataSize = nRows * nCols;
        int kernelSize = kRows * kCols;
        int fftSize = fftRows * fftCols;
        int cfftSize = fftRows * fftCols;
        int group_num = kernelCount / 2;

	// parameters for cufftPlanMany1d
        int idist = fftCols;
        int odist = fftCols;
        int inembed[] = {0};
        int onembed[] = {0};
        int n[1] = {fftCols};

	// parameter for cufftPlanMany2d
        int idist_2d = fftRows * fftCols;
        int odist_2d = fftRows * fftCols;
        int inembed_2d[] = {fftRows, fftCols};
        int onembed_2d[] = {fftRows, fftCols};
        int n_2d[2] = {fftRows, fftCols};

#if DEBUG
        for(int i = 0;i<nRows;i++){
            for(int j = 0;j<nCols;j++)
                printf("%f ", Signal[i * nCols + j]);
            printf("\n");
        }
        printf("\n");
        for(int i = 0;i<kRows;i++){
            for(int j = 0;j<kCols;j++)
                printf("%f ", Kernels[i * kCols + j]);
            printf("\n");
	}
        printf("\n");
#endif

        // device memory pointer
        float *d_signal = NULL;   // device storage of input signal
        float *d_kernels = NULL;  // device storage of kernels and final results
        cufftComplex *d_paddedSignal = NULL;
        cufftComplex *d_paddedKernels = NULL;  // device storage of kernels and final results
        float *d_results = NULL;
        cufftComplex *d_signal_fft;
        cufftComplex *d_kernels_fft;
        cufftComplex *d_w_results;
        
        cudaMalloc((void**)&d_signal, sizeof(float) * dataSize);
        cudaMalloc((void**)&d_kernels, sizeof(float) * kernelSize * kernelCount);
        cudaMalloc((void**)&d_paddedSignal, sizeof(cufftComplex) * fftSize);
        cudaMalloc((void**)&d_paddedKernels, sizeof(cufftComplex) * fftSize * group_num);
        cudaMalloc((void**)&d_results, sizeof(float) * fftSize * kernelCount);
        cudaMalloc((void**)&d_signal_fft, cfftSize * sizeof(cufftComplex));
        cudaMalloc((void**)&d_kernels_fft, cfftSize * group_num * sizeof(cufftComplex));
        cudaMalloc((void**)&d_w_results, cfftSize * group_num * sizeof(cufftComplex));

        // timer.restart();
        cudaMemset(d_signal, 0, dataSize * sizeof(float));
        cudaMemset(d_kernels, 0, kernelSize * kernelCount * sizeof(float));
        cudaMemset(d_paddedSignal, 0, fftSize * sizeof(cufftComplex));
        cudaMemset(d_paddedKernels, 0, fftSize * group_num * sizeof(cufftComplex));
        cudaMemset(d_results, 0, fftSize * kernelCount * sizeof(float));
        // timerMap["memset"].push_back(timer.elapsed());

        // timer.restart();
        cudaMemcpy(d_signal, Signal, dataSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernels, Kernels, kernelSize * kernelCount * sizeof(float), cudaMemcpyHostToDevice);
        //cudaMemcpy(Kernels, d_kernels, kernelSize * kernelCount * sizeof(float), cudaMemcpyDeviceToHost);
        //for(int i = 0;i<kernelSize;i++)
        //    printf("%f\n", Kernels[i]);
        //printf("\n");
        // timerMap["memcpy"].push_back(timer.elapsed());

        // padded signal
        dim3 threadBlock(16, 16, 4);
        dim3 dataBlockGrid(iDivUp(fftCols, threadBlock.x), iDivUp(fftRows, threadBlock.y), 1);
        dim3 kernelBlockGrid(iDivUp(fftCols, threadBlock.x), iDivUp(fftRows, threadBlock.y), iDivUp(group_num, threadBlock.z));
        //dim3 kernelBlockGrid(iDivUp(kCols, threadBlock.x), iDivUp(kRows, threadBlock.y), iDivUp(group_num, threadBlock.z));
        padDataWithZerosComplex<<<dataBlockGrid, threadBlock>>>(d_paddedSignal, d_signal, fftCols, fftRows, nCols, nRows, 1); 
        padKernelWithZerosComplex<<<kernelBlockGrid, threadBlock>>>(d_paddedKernels, d_kernels, fftCols, fftRows, kCols, kRows, group_num);

#if DEBUG
        CudaCheckError();
        cufftComplex *h_paddedSignal = (cufftComplex*)malloc(sizeof(cufftComplex) * fftSize);
        cufftComplex *h_paddedKernels = (cufftComplex*)malloc(sizeof(cufftComplex) * fftSize * group_num);

        cudaMemcpy(h_paddedSignal, d_paddedSignal, sizeof(cufftComplex) * fftSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_paddedKernels, d_paddedKernels, sizeof(cufftComplex) * fftSize * group_num, cudaMemcpyDeviceToHost);

        for(int i = 0;i<fftRows;i++){
            for(int j = 0;j<fftCols;j++)
            	printf("%f+%fi ", h_paddedSignal[i * fftCols + j].x, h_paddedSignal[i * fftCols + j].y);
            printf("\n");
	}
        printf("\n");
        for(int i = 0;i<fftRows;i++){
            for(int j = 0;j<fftCols;j++)
            	printf("%f+%fi ", h_paddedKernels[i * fftCols + j].x, h_paddedKernels[i * fftCols + j].y);
            printf("\n");
	}
        printf("\n");
#endif
        cufftPlan2d(&fwplan_signal, fftRows, fftCols, CUFFT_C2C);
        cufftPlanMany(&fwplan_kernels, 2, n_2d, inembed_2d, 1, idist_2d, onembed_2d, 1, odist_2d, CUFFT_C2C, group_num);
        cufftPlanMany(&bwplan, 2, n_2d, onembed_2d, 1, odist_2d, inembed_2d, 1, idist_2d, CUFFT_C2C, group_num);

	// create 1d plan
        cufftPlanMany(&plan1d, 1, n, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, group_num * fftCols * 3 / 2);

#if DEBUG
        printf("finish plan construction.\n");
#endif
	// warm up
	{

            cufftExecC2C(fwplan_signal, d_paddedSignal, d_signal_fft, CUFFT_FORWARD);
            //cufftExecC2C(fwplan_kernels, d_paddedKernels, d_kernels_fft, CUFFT_FORWARD);
            //fast_fft(d_paddedSignal, 1, fftCols, fftRows, kCols, kRows, d_signal_fft, true);
            fast_fft(d_paddedKernels, group_num, fftCols, fftRows, kCols, kRows, d_kernels_fft, true);

            dim3 dotBlock(32, 32);
            dim3 dotGrid(fftCols / 32, fftCols / 32, 8);
			dotProduct<<<dotGrid, dotBlock>>>(d_signal_fft, fftCols, fftRows, 1.0 / fftSize, group_num/8, d_kernels_fft);

            // thrust::device_ptr<cufftComplex> thr_signal_fft(d_signal_fft);
            // thrust::device_ptr<cufftComplex> thr_kernel_fft(d_kernels_fft);
            // //thrust::transform(thr_signal_fft, thr_signal_fft + group_num * cfftSize, thr_kernel_fft, thr_kernel_fft, complexMultiplier(1.0 / fftSize));
            // for(int i = 0 ;i < group_num; i++){
            //     thrust::transform(thr_signal_fft, thr_signal_fft + cfftSize, thr_kernel_fft + i * cfftSize, thr_kernel_fft + i * cfftSize,
            //                       complexMultiplier(1.0 / fftSize));
            //     // timerMap["multiply"].push_back(timer.elapsed());
            // }

#if VERBOSE
            size_t freeMem, totalMem;
            cudaMemGetInfo(&freeMem, &totalMem);
            std::cout << "Free: " << freeMem << ", Total: " << totalMem << std::endl;
#endif

            //cudaMalloc((void**)&d_conj_w_results, cfftSize * group_num * sizeof(cufftComplex));
            // timer.restart();
            cufftExecC2C(bwplan, d_kernels_fft, d_w_results, CUFFT_INVERSE);
            //fast_fft(d_kernels_fft, group_num, fftCols, fftRows, d_w_results, false);
	}


        cudaDeviceSynchronize();

	// timing record
	int fft_time = 0;
	int dot_product_time = 0;
	int ifft_time = 0;

	    //int duplicate = 8;
		//cufftComplex* d_double_signal_fft;
        //cudaMalloc((void**)&d_double_signal_fft, cfftSize * duplicate * sizeof(cufftComplex));
        int num_repeats = NUM_ITERATIONS;
        for (int i = 0; i < num_repeats; ++i) 
	{
            auto start = std::chrono::steady_clock::now();
            cufftExecC2C(fwplan_signal, d_paddedSignal, d_signal_fft, CUFFT_FORWARD);
            //cufftExecC2C(fwplan_kernels, d_paddedKernels, d_kernels_fft, CUFFT_FORWARD);
            //fast_fft(d_paddedSignal, 1, fftCols, fftRows, d_signal_fft, true);
            fast_fft(d_paddedKernels, group_num, fftCols, fftRows, kCols, kRows, d_kernels_fft, true);
            auto end = std::chrono::steady_clock::now();
            fft_time += static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() );

            // timerMap["forward"].push_back(timer.elapsed());

            // timer.restart();

	    start = std::chrono::steady_clock::now();
            dim3 dotBlock(32, 32);
            dim3 dotGrid(fftCols / 32, fftCols / 32, 8);
			dotProduct<<<dotGrid, dotBlock>>>(d_signal_fft, fftCols, fftRows, 1.0 / fftSize, group_num/8, d_kernels_fft);
			//for(int c = 0; c < duplicate;c++)
            //    cudaMemcpy(d_double_signal_fft + c * fftSize, d_signal_fft, sizeof(cufftComplex) * fftSize, cudaMemcpyDeviceToDevice);
            //thrust::device_ptr<cufftComplex> thr_signal_fft(d_double_signal_fft);
            //thrust::device_ptr<cufftComplex> thr_kernel_fft(d_kernels_fft);
            ////thrust::transform(thr_signal_fft, thr_signal_fft + group_num * cfftSize, thr_kernel_fft, thr_kernel_fft, complexMultiplier(1.0 / fftSize));
            //for(int i = 0 ;i < group_num / duplicate; i++){
            //    thrust::transform(thr_signal_fft, thr_signal_fft + duplicate * fftSize, thr_kernel_fft + i * duplicate * fftSize, thr_kernel_fft + i * duplicate * fftSize,
            //                      complexMultiplier(1.0 / fftSize));
            //    // timerMap["multiply"].push_back(timer.elapsed());
            //}
            cudaDeviceSynchronize();
	    end = std::chrono::steady_clock::now();
            dot_product_time += static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() );

#if VERBOSE
            size_t freeMem, totalMem;
            cudaMemGetInfo(&freeMem, &totalMem);
            std::cout << "Free: " << freeMem << ", Total: " << totalMem << std::endl;
#endif

            //cudaMalloc((void**)&d_conj_w_results, cfftSize * group_num * sizeof(cufftComplex));
            // timer.restart();
	    start = std::chrono::steady_clock::now();
            //fast_fft(d_kernels_fft, group_num, fftCols, fftRows, d_w_results, false);
            cufftExecC2C(bwplan, d_kernels_fft, d_w_results, CUFFT_INVERSE);
            cudaDeviceSynchronize();
	    end = std::chrono::steady_clock::now();
            ifft_time += static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() );

	}

        int avg_fft_time = fft_time / num_repeats;
        int avg_dot_product_time = dot_product_time / num_repeats;
        int avg_ifft_time = ifft_time / num_repeats;
	int avg_total_time = avg_fft_time + avg_dot_product_time + avg_ifft_time;
	printf("turbo_fft_gpu_v3(us)----fft:%d, dot_prod:%d, ifft:%d, total:%d\n", avg_fft_time, avg_dot_product_time, avg_ifft_time, avg_total_time);

        // split complex results
        SplitComplex<<<kernelBlockGrid, threadBlock>>>(d_results, d_w_results, fftCols, fftRows, group_num);
        //ConjComplex<<<kernelBlockGrid, threadBlock>>>(d_conj_w_results, d_w_results, fftCols, fftRows, group_num);
        //cufftExecZ2Z(*bwplan, d_kernels_fft, d_kernels, CUFFT_INVERSE);
        // timerMap["backward"].push_back(timer.elapsed());

        // timer.restart();
        cudaMemcpy(results, d_results, kernelCount * fftSize * sizeof(float), cudaMemcpyDeviceToHost);
       
        // timerMap["copy_back"].push_back(timer.elapsed());
        // timerMap["total_time"].push_back(globalTimer.elapsed());

        // free GPU memory
        cudaFree(d_signal);
        cudaFree(d_kernels);
        cudaFree(d_paddedSignal);
        cudaFree(d_paddedKernels);
        cudaFree(d_results);
        cudaFree(d_signal_fft);
        cudaFree(d_kernels_fft);
        cudaFree(d_w_results);
        // free GPU memory for qdVectors
        cudaFree(d_w_results);
        cudaFree(d_qdVectors);

    }
    // direct fft for complex kernels
    void direct_fft(float * Kernels, int kernelCount, 
                             int nRows, int nCols, int kRows, int kCols, cufftComplex * results)
    {

        cufftHandle fwplan_kernels, bwplan; 

	int group_num = kernelCount / 2;
        int fftRows = nRows + kRows - 1;
        int fftCols = nCols + kCols - 1;
        int dataSize = nRows * nCols;
        int kernelSize = kRows * kCols;
        int fftSize = fftRows * fftCols;

	// parameters for cufftPlanMany1d
        int idist = fftCols * fftRows;
        int odist = fftCols * fftRows;
        int inembed[] = {fftCols, fftRows};
        int onembed[] = {fftCols, fftRows};
        int n[2] = {fftCols, fftRows};

#if DEBUG
        for(int i = 0;i<kRows;i++){
            for(int j = 0;j<kCols;j++)
                printf("%f ", Kernels[i * kCols + j]);
            printf("\n");
	}
        printf("\n");
#endif

        // device memory pointer
        float *d_kernels = NULL;  // device storage of kernels and final results
        cufftComplex *d_paddedKernels = NULL;  // device storage of kernels and final results
        cufftComplex *d_kernels_fft;
        cufftComplex *d_results;
        
        cudaMalloc((void**)&d_kernels, sizeof(float) * kernelSize * kernelCount);
        cudaMalloc((void**)&d_paddedKernels, sizeof(cufftComplex) * fftSize * group_num);
        cudaMalloc((void**)&d_kernels_fft, sizeof(cufftComplex) * fftSize * group_num);
        cudaMalloc((void**)&d_results, fftSize * group_num * sizeof(cufftComplex));

        // timer.restart();
        cudaMemset(d_kernels, 0, kernelSize * kernelCount * sizeof(float));
        cudaMemset(d_paddedKernels, 0, fftSize * group_num * sizeof(cufftComplex));
        cudaMemset(d_kernels_fft, 0, fftSize * group_num * sizeof(cufftComplex));
        cudaMemset(d_results, 0, fftSize * group_num * sizeof(cufftComplex));
        // timerMap["memset"].push_back(timer.elapsed());

        cudaMemcpy(d_kernels, Kernels, kernelSize * kernelCount * sizeof(float), cudaMemcpyHostToDevice);
        // padded signal
        dim3 threadBlock(16, 16, 4);
        dim3 dataBlockGrid(iDivUp(fftCols, threadBlock.x), iDivUp(fftRows, threadBlock.y), 1);
        dim3 kernelBlockGrid(iDivUp(fftCols, threadBlock.x), iDivUp(fftRows, threadBlock.y), iDivUp(group_num, threadBlock.z));
        padKernelWithZerosComplex<<<kernelBlockGrid, threadBlock>>>(d_paddedKernels, d_kernels, fftCols, fftRows, kCols, kRows, group_num);

#if DEBUG
        CudaCheckError();
        cufftComplex *h_paddedKernels = (cufftComplex*)malloc(sizeof(cufftComplex) * fftSize * group_num);

        cudaMemcpy(h_paddedKernels, d_paddedKernels, sizeof(cufftComplex) * fftSize * group_num, cudaMemcpyDeviceToHost);

        for(int i = 0;i<fftRows;i++){
            for(int j = 0;j<fftCols;j++)
            	printf("%f+%fi ", h_paddedKernels[i * fftCols + j].x, h_paddedKernels[i * fftCols + j].y);
            printf("\n");
	}
        printf("\n");
#endif

	// create 2d plan
        cufftPlanMany(&fwplan_kernels, 2, n, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, group_num);
        cufftExecC2C(fwplan_kernels, d_paddedKernels, d_kernels_fft, CUFFT_INVERSE);
        cudaDeviceSynchronize();
        cudaMemcpy(results, d_kernels_fft, group_num * fftSize * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
        //cudaMemcpy(results, d_results, kernelCount * fftSize * sizeof(float), cudaMemcpyDeviceToHost);

#if DEBUG
        for(int i = 0;i<kRows;i++){
            for(int j = 0;j<kCols;j++)
                printf("%f ", results[i * kCols + j].y);
            printf("\n");
	}
        printf("\n");
#endif

        // free GPU memory
        cudaFree(d_kernels);
        cudaFree(d_paddedKernels);
        cudaFree(d_results);
        cudaFree(d_kernels_fft);

    }

    // test fft with QD algorithm
    void qd_fft(float * Kernels, int kernelCount, 
                             int nRows, int nCols, int kRows, int kCols, cufftComplex * results)
    {

        cufftHandle fwplan_kernels, bwplan; 

	int group_num = kernelCount / 2;
        int fftRows = nRows + kRows - 1;
        int fftCols = nCols + kCols - 1;
        int dataSize = nRows * nCols;
        int kernelSize = kRows * kCols;
        int fftSize = fftRows * fftCols;
        int cfftSize = fftRows * fftCols;

	// parameters for cufftPlanMany1d
        int idist = fftCols;
        int odist = fftCols;
        int inembed[] = {0};
        int onembed[] = {0};
        int n[1] = {fftCols};

#if DEBUG
        for(int i = 0;i<kRows;i++){
            for(int j = 0;j<kCols;j++)
                printf("%f ", Kernels[i * kCols + j]);
            printf("\n");
	}
        printf("\n");
#endif

        // device memory pointer
        float *d_kernels = NULL;  // device storage of kernels and final results
        cufftComplex *d_paddedKernels = NULL;  // device storage of kernels and final results
        cufftComplex *d_w_results = NULL;
        cufftComplex *d_results = NULL;
        
        cudaMalloc((void**)&d_kernels, sizeof(float) * kernelSize * kernelCount);
        cudaMalloc((void**)&d_paddedKernels, sizeof(cufftComplex) * fftSize * group_num);
        cudaMalloc((void**)&d_w_results, fftSize * 3 / 2 * group_num * sizeof(cufftComplex));
        cudaMalloc((void**)&d_results, fftSize * group_num * sizeof(cufftComplex));

        // timer.restart();
        cudaMemset(d_kernels, 0, kernelSize * kernelCount * sizeof(float));
        cudaMemset(d_paddedKernels, 0, fftSize * group_num * sizeof(cufftComplex));
        cudaMemset(d_w_results, 0, fftSize * 3 / 2 * group_num * sizeof(cufftComplex));
        // timerMap["memset"].push_back(timer.elapsed());

        cudaMemcpy(d_kernels, Kernels, kernelSize * kernelCount * sizeof(float), cudaMemcpyHostToDevice);
#if DEBUG
        printf("begin padding.\n");
#endif

        // padded signal
        dim3 threadBlock(16, 16, 4);
        dim3 dataBlockGrid(iDivUp(fftCols, threadBlock.x), iDivUp(fftRows, threadBlock.y), 1);
        dim3 kernelBlockGrid(iDivUp(fftCols, threadBlock.x), iDivUp(fftRows, threadBlock.y), iDivUp(group_num, threadBlock.z));
        padKernelWithZerosComplex<<<kernelBlockGrid, threadBlock>>>(d_paddedKernels, d_kernels, fftCols, fftRows, kCols, kRows, group_num);

	fast_fft(d_paddedKernels, group_num, fftCols, fftRows, kCols, kRows, d_results, false);
	cudaMemcpy(results, d_results, sizeof(cufftComplex) * fftSize * group_num, cudaMemcpyDeviceToHost);
//#if DEBUG
//        printf("begin qd fft.\n");
//#endif
//
//	// cpu version of QD fft
//        cufftComplex *h_paddedKernels = (cufftComplex*)malloc(sizeof(cufftComplex) * fftSize * group_num);  // padded kernels
//        cudaMemcpy(h_paddedKernels, d_paddedKernels, sizeof(cufftComplex) * fftSize * group_num, cudaMemcpyDeviceToHost);
//
//#if DEBUG
//        for(int i = 0;i<fftRows;i++){
//            for(int j = 0;j<fftCols;j++)
//            	printf("%f+%fi ", h_paddedKernels[i * fftCols + j].x, h_paddedKernels[i * fftCols + j].y);
//            printf("\n");
//	}
//        printf("\n");
//#endif
//
//	// initilize p and s
//	int* p = new int[fftCols + fftCols / 2];
//	int* s = new int[fftCols + fftCols / 2];
//	for(int i = 0 ; i < fftCols; i ++){
//	    p[i] = 1;
//	    s[i] = i;
//	}
//	for(int i = fftCols; i < fftCols + fftCols / 2;i++){
//	    p[i] = (i - fftCols) * 2;
//	    s[i] = 1;
//	}
//
//#if DEBUG
//        printf("finish initilization of (p, s).\n");
//	for(int i = 0 ; i < fftCols + fftCols / 2;i++)
//	    printf("%d: p:%d, s:%d.\n", i, p[i], s[i]);
//#endif
//
//	// initilize indices of get data, 1.5 * N * 2N^2, indices of out data, 
//	int N = fftCols;
//	int* input_xs = new int[N * 3 / 2 * (N * N)];
//	int* input_ys = new int[N * 3 / 2 * (N * N)];
//	int* output_xs = new int[N * 3 / 2 * N];
//	int* output_ys = new int[N * 3 / 2 * N];
//
//        for(int i = 0; i < fftCols + fftCols/2;i++){
//	    for(int t = 0 ; t < fftCols;t++){
//		output_xs[i * fftCols + t] = (t * p[i]) % fftCols;
//		output_ys[i * fftCols + t] = (t * s[i]) % fftCols;
//	    }
//	}
//
//	for(int i = 0; i < fftCols + fftCols/2;i++)
//	    for(int t = 0; t < fftCols; t++){
//		int offset = i * fftSize + t * fftCols;
//	        find_input_index(p[i], s[i], t, fftCols, input_xs + offset, input_ys + offset);
//
////#if DEBUG
////		for(int j = 0; j < fftCols;j++)
////		    printf("%d, %d.\n", input_xs[offset + j], input_ys[offset + j]);
////		getchar();
////#endif
//	    }
//        
//#if DEBUG
//        printf("finish indices search.\n");
//#endif
//
//	cufftComplex *h_qdVectors = (cufftComplex*)malloc(sizeof(cufftComplex) * fftSize * 3 / 2 * group_num);
//        memset(h_qdVectors, 0, sizeof(cufftComplex) * fftSize * 3 / 2 * group_num);
//	// construct 1d vectors for 1d-fft
//	for(int k = 0; k < group_num;k++){
//	    int kOffset = k * fftSize * 3 / 2;
//            for(int i = 0; i < fftCols + fftCols/2;i++)
//	        for(int t = 0; t < fftCols; t++)
//	            for(int j = 0; j < fftCols;j++){
//	    	        int offset = i * fftSize + t * fftCols + j;
//	    	        int idx_x = input_xs[offset];
//	    	        int idx_y = input_ys[offset];
//	                h_qdVectors[kOffset + i * fftCols + t].x += h_paddedKernels[k * fftSize + idx_x * fftCols + idx_y].x;
//	                h_qdVectors[kOffset + i * fftCols + t].y += h_paddedKernels[k * fftSize + idx_x * fftCols + idx_y].y;
//	    	}
//	}
//
//#if DEBUG
//        printf("finish vector construction.\n");
//#endif
//
//        cufftComplex *d_qdVectors;
//        cudaMalloc((void**)&d_qdVectors, sizeof(cufftComplex) * fftSize * 3 / 2 * group_num);
//	cudaMemcpy(d_qdVectors, h_qdVectors, sizeof(cufftComplex) * fftSize * 3 / 2 * group_num, cudaMemcpyHostToDevice);
//	// create 1d plan
//        cufftPlanMany(&fwplan_kernels, 1, n, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, group_num * fftCols * 3 / 2);
//
//#if DEBUG
//        printf("finish plan construction.\n");
//#endif
//
//        cufftExecC2C(fwplan_kernels, d_qdVectors, d_w_results, CUFFT_FORWARD);
//
//        cudaDeviceSynchronize();
//
//#if DEBUG
//        printf("finish fft.\n");
//#endif
//
//	cufftComplex* h_w_results = (cufftComplex*)malloc(group_num * fftSize * 3 / 2 * sizeof(cufftComplex));
//        cudaMemcpy(h_w_results, d_w_results, group_num * fftSize * 3 / 2 * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
//
//        // re-arrange results
//	for(int k = 0; k < group_num;k++){
//	    int kOffset = k * fftSize * 3 / 2;
//            for(int i = 0; i < fftCols + fftCols/2;i++)
//	        for(int t = 0; t < fftCols; t++){
//		    int idx_x = output_xs[i * fftCols + t];
//		    int idx_y = output_ys[i * fftCols + t];
//		    results[k * fftSize + idx_x * fftCols + idx_y].x = h_w_results[kOffset + i * fftCols + t].x;
//		    results[k * fftSize + idx_x * fftCols + idx_y].y = h_w_results[kOffset + i * fftCols + t].y;
//		}
//		
//	}
//
//        // free GPU memory
//        cudaFree(d_kernels);
//        cudaFree(d_paddedKernels);
//        cudaFree(d_w_results);
//        cudaFree(d_qdVectors);
#if DEBUG
        CudaCheckError();
        for(int i = 0;i<fftRows;i++){
            for(int j = 0;j<fftCols;j++)
            	printf("%f+%fi ", results[i * fftCols + j].x, results[i * fftCols + j].y);
            printf("\n");
	}
        printf("\n");
#endif

    }
//}
