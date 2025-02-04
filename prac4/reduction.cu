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

/*

To perform multi-block reduction:
- Step 1:
  - use temp memory / number of blocks (32 can be optimal) 
  - Add using per warp shuffle 
  - Write result -> first thread of each warp adds sum to its block's position in (per block) shared memory (of size e.g. 32 floats)

- Step 2:
  - first warp add shared memory (of max size 32) using shuffle
  - Write result -> Use atomicAdd to add to globalAccumulator

*/

__global__ void reduction(float *g_odata, float *g_idata)
{
    // dynamically allocated shared memory

    extern  __shared__  float temp[];

    int tid = threadIdx.x;

    // first, each thread loads data into shared memory

    temp[tid] = g_idata[tid + blockDim.x * blockIdx.x];

    // extension for non pow2 threads
    unsigned int round_down_2 =  (1 << (32 - __clz(blockDim.x-1)))>>1;  
    unsigned int remaining_elements = blockDim.x-round_down_2;
    
    __syncthreads();  // ensure data loaded
    if (tid<remaining_elements) temp[tid] += temp[tid+round_down_2];
    // if(tid==0){
    //   printf("GPU:: Remaining elements: %u\n",remaining_elements);
    //   printf("GPU:: round_down_2: %u\n",round_down_2);
    // }

    // next, we perform binary tree reduction

    // for (int d=blockDim.x/2; d>0; d=d/2) { //code when pow2 threads
    for (int d=round_down_2/2; d>0; d=d/2) {
      __syncthreads();  // ensure previous step completed 
      if (tid<d)  temp[tid] += temp[tid+d];
    }

    // finally, first thread puts result into global memory
    // if (tid==0) g_odata[0] = temp[0];

    // finally, add numbers to g_odata
    if (tid==0) atomicAdd(g_odata,temp[tid]);
}


__global__ void reduction_shuffle(float *g_odata, float *g_idata)
{
    // dynamically allocated shared memory
    extern  __shared__  float temp[];

    int tid = threadIdx.x;

    // first, each thread loads data into register
    float temp_sum;
    temp_sum = g_idata[tid + blockDim.x * blockIdx.x];

    // next, we perform shuffle reduction per warp (for all warps)
    for (int i=1; i<32; i=2*i)
      temp_sum += __shfl_xor_sync((unsigned int)-1, temp_sum, i);

    //next thread 1 of each warp writes to shared memory
    if (tid%warpSize==0){
      temp[tid/warpSize]=temp_sum;
    }
    __syncthreads();  // wait for shared memory write

    //finally first warp reduces all shared memory
    if (tid<warpSize){
      temp_sum = 0.0f;
      if (tid<blockDim.x/warpSize) temp_sum = temp[tid];

      for (int i=1; i<32; i=2*i)
        temp_sum += __shfl_xor_sync((unsigned int)-1, temp_sum, i);
    }

    // finally, add numbers to g_odata
    if (tid==0) atomicAdd(g_odata,temp_sum);
}



__global__ void reduction_suffle_gold(float *g_odata, float *g_idata)
{
    // shared memory

    __shared__ float temp[32];

    int   tid = threadIdx.x;
    float val = g_idata[tid + blockIdx.x*blockDim.x];

    // first, do reduction within each warp

    for (int i=1; i<32; i=2*i)
      val += __shfl_xor_sync((unsigned int)-1, val, i);

    // put warp sums into shared memory, then read back into first warp

    if (tid%32==0) temp[tid/32] = val;

    __syncthreads();

    if (tid<32) {
      val = 0.0f;
      if (tid<blockDim.x/32) val = temp[tid];

    // second, do final reduction within first warp

      for (int i=1; i<32; i=2*i)
        val += __shfl_xor_sync((unsigned int)-1, val, i);

    // finally, first thread atomically adds result to global sum

      if (tid==0) atomicAdd(g_odata, val);
    }
}

////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{

  bool run_shuffle = true;
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
  num_threads  = 512;
  num_elements = num_blocks*num_threads;
  mem_size     = sizeof(float) * num_elements;

  if (run_shuffle){
    printf("Running Shuffle");
  }
  else{
    printf("Running Binary Tree Reduction");
  }
  printf("Blocks: %d, Threads: %d\n",num_blocks,num_threads);

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
  float sum2 = 0.0f;
  cudaEventRecord(start);
  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(d_odata, &sum2, sizeof(float),
                              cudaMemcpyHostToDevice) );
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("Memcopy took: %.5f (ms) \n\n", milli);  

  // execute the kernel

  shared_mem_size = sizeof(float) * num_threads;

  cudaEventRecord(start);
  // reduction<<<num_blocks,num_threads,shared_mem_size>>>(d_odata,d_idata);
  // reduction_shuffle<<<num_blocks,num_threads,32>>>(d_odata,d_idata);
  reduction_suffle_gold<<<num_blocks,num_threads>>>(d_odata,d_idata);
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
