
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcurand

NVCCFLAGS	:= -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

all:	prac2 prac2_device

run1:
	sbatch prac2.sh 

run2:
	sbatch prac22.sh

prac2:	prac2.cu Makefile
	nvcc prac2.cu -o prac2 $(INC) $(NVCCFLAGS) $(LIB)

prac22:	prac22.cu Makefile
	nvcc prac22.cu -o prac22 $(INC) $(NVCCFLAGS) $(LIB)

prac2_device:	prac2_device.cu Makefile
	nvcc prac2_device.cu -o prac2_device $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f prac2 prac2_device prac22
