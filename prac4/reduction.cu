////////////////////////////////////////////////////////////////////////
//
// Practical 4 -- initial code for shared memory reduction for 
//                a single block which is a power of two in size
//
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// CPU routine
////////////////////////////////////////////////////////////////////////

float reduction_gold(float* idata, int len) 
{
  float sum = 0.0f;
  for(int i=0; i<len; i++) sum += idata[i];

  return sum;
}

////////////////////////////////////////////////////////////////////////
// GPU routine
////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_odata, float *g_idata)
{
    // dynamically allocated shared memory

    extern  __shared__  float temp[];

    int tid = threadIdx.x;

    // first, each thread loads data into shared memory

    temp[tid] = g_idata[tid];

    // extension for non pow2 threads
    unsigned int round_down_2 =  (1 << (32 - __clz(blockDim.x-1)))>>1;  
    unsigned int remaining_elements = blockDim.x-round_down_2;
    __syncthreads();  // ensure previous step completed 
    if (tid<remaining_elements) temp[tid] += temp[tid+round_down_2];
    if(tid==0){
      printf("GPU:: Remaining elements: %u\n",remaining_elements);
      printf("GPU:: round_down_2: %u\n",round_down_2);
    }

    // next, we perform binary tree reduction

    // for (int d=blockDim.x/2; d>0; d=d/2) { //code when pow2 threads
    for (int d=round_down_2/2; d>0; d=d/2) {
      __syncthreads();  // ensure previous step completed 
      if (tid<d)  temp[tid] += temp[tid+d];
    }

    // finally, first thread puts result into global memory
    if (tid==0) g_odata[0] = temp[0];
}


////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_blocks, num_threads, num_elements, mem_size, shared_mem_size;

  float *h_data, *d_idata, *d_odata;

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing
  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  // General params
  num_blocks   = 1;  // start with only 1 thread block
  num_threads  = 512+64;
  num_elements = num_blocks*num_threads;
  mem_size     = sizeof(float) * num_elements;



  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 10

  h_data = (float*) malloc(mem_size);

  for(int i = 0; i < num_elements; i++) 
    h_data[i] = floorf(10.0f*(rand()/(float)RAND_MAX));

  // compute reference solution
  cudaEventRecord(start);
  float sum = reduction_gold(h_data, num_elements);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("CPU took: %.5f (ms) \n\n", milli);  

  // allocate device memory input and output arrays

  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_odata, sizeof(float)) );

  // copy host memory to device input array
  cudaEventRecord(start);
  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice) );
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("Memcopy took: %.5f (ms) \n\n", milli);  

  // execute the kernel

  shared_mem_size = sizeof(float) * num_threads;

  cudaEventRecord(start);
  reduction<<<num_blocks,num_threads,shared_mem_size>>>(d_odata,d_idata);
  getLastCudaError("reduction kernel execution failed");
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("Kernel took: %.5f (ms) \n\n", milli);  
  
  
  // copy result from device to host
  cudaEventRecord(start);
  checkCudaErrors( cudaMemcpy(h_data, d_odata, sizeof(float),
                              cudaMemcpyDeviceToHost) );
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("Memcopy took: %.5f (ms) \n\n", milli);  
  // check results

  printf("reduction error = %f\n",h_data[0]-sum);

  // cleanup memory

  free(h_data);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_odata) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
