//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>


//
// kernel routine
// 

// __global__ void my_first_kernel(float *x)
// {
//   int tid = threadIdx.x + blockDim.x*blockIdx.x;
//   float y = 1/0;
//   int z = -1/0;

//   if (threadIdx.x == 1){
//     printf("y is %f, z is %d\n",y,z);
//   }

//   x[tid] = (float) y;
//   // x[tid+10000000] = (float) y;

// }


__global__ void add_kernel(float *x, float*y, float*z)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  z[tid] = x[tid]+y[tid];

}

__global__ void init_kernel(float *x, float*y, float*z)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] = (float) tid;
  y[tid] = 2*(float) tid;

}

//
// main code
//

int main(int argc, char **argv)
{
  float *h_x, *d_x;
  float *h_y, *d_y;
  float *h_z, *d_z;
  int   nblocks, nthreads, nsize, n; 

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array

  h_x = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_x, nsize*sizeof(float)));

  h_y = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_y, nsize*sizeof(float)));

  h_z = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_z, nsize*sizeof(float)));

  // Init on host

  // for (n=0; n<nsize; n++) h_x[n] = n;
  // for (n=0; n<nsize; n++) h_y[n] = n;

  // copy init to Cuda


  // cudaMemcpy(d_x,h_x,nsize*sizeof(float),cudaMemcpyHostToDevice);
  // cudaMemcpy(d_y,h_y,nsize*sizeof(float),cudaMemcpyHostToDevice);


  // execute kernel (this call is non blocking)

  // my_first_kernel<<<nblocks,nthreads>>>(d_x);
  // my_first_custom_kernel<<<nblocks,nthreads>>>(d_x,d_y,d_z);

  init_kernel<<<nblocks,nthreads>>>(d_x,d_y,d_z);
  getLastCudaError("init_kernel execution failed\n");

  add_kernel<<<nblocks,nthreads>>>(d_x,d_y,d_z);
  getLastCudaError("add_kernel execution failed\n");

  
  // copy back results and print them out

  checkCudaErrors(cudaMemcpy(h_z,d_z,nsize*sizeof(float),cudaMemcpyDeviceToHost));

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_z[n]);

  // free memory 

  cudaFree(d_x);
  free(h_x);

  cudaFree(d_y);
  free(h_y);

  cudaFree(d_z);
  free(h_z);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
