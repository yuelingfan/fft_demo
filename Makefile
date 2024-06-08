FLAGS = -std=c++11 -lcublas -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -res-usage -lcudart -lfftw3 -lcufft -lineinfo -Xcompiler -fopenmp
OBJ= main main_conv
INCLUDE=-I./include

main: main.cpp tcfft_half.cu cufft.cpp tcfft_half_2d.cu tensorcore_fft_conv2d.cpp tensorcore_fft_conv2d_kernel.cu fftconvolve2d.cu
	nvcc $^ -o $@ $(FLAGS) $(INCLUDE)

main_conv : main_2d.cpp tensorcore_fft_conv2d.cpp tensorcore_fft_conv2d_kernel.cu fftconvolve2d.cu
	nvcc $^ -o $@ $(FLAGS) $(INCLUDE)

clean :
	rm -f $(OBJ)