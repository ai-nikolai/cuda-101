
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcufft

NVCCFLAGS	:= -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

both:		simpleCUFFT tensorCUBLAS

tensorCUBLAS:	cuBLAS_Tensors.cu Makefile
		nvcc cuBLAS_Tensors.cu -o tensorCUBLAS $(INC) $(LIB)

simpleCUFFT:	simpleCUFFT.cu Makefile
		nvcc simpleCUFFT.cu -o simpleCUFFT $(INC) $(NVCCFLAGS) $(LIB)

clean:
		rm -f tensorCUBLAS simpleCUFFT
