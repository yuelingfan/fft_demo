/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

// A simple timer class
// use CUDA's high-resolution timers when possible
#include <cuda_runtime_api.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <sys/time.h>

#define TIMING

/********************/
/* CUDA ERROR CHECK */
/********************/
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line )
{
//#ifdef CUDA_ERROR_CHECK
    printf("CUDA Error Check enabled.\n");
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
//#endif

    return;
}


static void cuda_safe_call(cudaError_t error, const std::string& message = "")
{
    if (error)
        throw thrust::system_error(error, thrust::cuda_category(), message);
}

struct timer
{
    cudaEvent_t start;
    cudaEvent_t end;

    timer(void)
    {
#ifdef TIMING
        cuda_safe_call(cudaEventCreate(&start));
        cuda_safe_call(cudaEventCreate(&end));
        restart();
#endif

    }

    ~timer(void)
    {
#ifdef TIMING
        cuda_safe_call(cudaEventDestroy(start));
        cuda_safe_call(cudaEventDestroy(end));
#endif
    }

    void restart(void)
    {
#ifdef TIMING
        cuda_safe_call(cudaEventRecord(start, 0));
#endif
    }
    // In ms
    double elapsed(void)
    {
#ifdef TIMING
        cuda_safe_call(cudaEventRecord(end, 0));
        cuda_safe_call(cudaEventSynchronize(end));

        // double ms_elapsed;
        float ms_elapsed;
        cuda_safe_call(cudaEventElapsedTime(&ms_elapsed, start, end));
        return ms_elapsed;
#else
        return 0.0;
#endif
    }

    double epsilon(void)
    {
        return 0.5e-6;
    }
};


   static inline double time_diff(timespec const &end, timespec const &begin)
   {
#ifdef TIMING
      double result;

      result = end.tv_sec - begin.tv_sec;
      result += (end.tv_nsec - begin.tv_nsec) / (double) 1000000000;

      return result;
#else
      return 0;
#endif
   }

   static inline void get_time(timespec &ts)
   {

#ifdef TIMING
      struct timeval tv;
      gettimeofday(&tv, NULL);
      ts.tv_sec = tv.tv_sec;
      ts.tv_nsec = tv.tv_usec * 1000;
#endif
   }

   static inline timespec get_time()
   {
      timespec t;
#ifdef TIMING
      get_time(t);
#endif
      return t;
   }

   static inline double time_elapsed(timespec const &begin)
   {
#ifdef TIMING
      timespec now;
      get_time(now);
      return time_diff(now, begin);
#else
      return 0;
#endif
   }

   static inline void print_time(char const *prompt, timespec const &begin,
                                 timespec const &end)
   {
#ifdef TIMING
      printf("%s : %.3f\n", prompt, time_diff(end, begin));
      // dprintf("%s : %.3f\n", prompt, time_diff(end, begin));
#endif
   }

   static inline void print_time(char const *prompt, double diff)
   {
#ifdef TIMING
      printf("%s : %.3f\n", prompt, diff);
      // dprintf("%s : %.3f\n", prompt, diff);
#endif
   }

   static inline void print_time_elapsed(char const *prompt,
                                         timespec const &begin)
   {
#ifdef TIMING
      // dprintf("%s : %.3f\n", prompt, time_elapsed(begin));
      printf("%s : %.3f\n", prompt, time_elapsed(begin));
#endif
   }
