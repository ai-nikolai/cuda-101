//
// a test of 3 different methods for rounding an integer
// up to the nearest power of 2 for use in Practical 4;
// this code can be compiled and run on a CPU to test it
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main()
{

  int m1, m2, m3;

  unsigned int test = 0x80000000, mask=1;

  mask = mask<< 31;

  printf("sizeof unsigned int:%d\n",sizeof(test));
  printf("Value of test:%u\nAnd:%x\n",test, test);
  printf("Value of mask:%u\nAnd:%x\n",mask, mask);


  if (mask==test){
    printf("We won!\n");
  }
   
  for (int n=2; n<17; n++){

    for (m1=1; m1<n; m1=2*m1) {} 

    m2 = n-1;
    m2 = m2 | (m2>>1);
    m2 = m2 | (m2>>2);
    m2 = m2 | (m2>>4);
    m2 = m2 | (m2>>8);  // this handles up to 16-bit integers
    // m2 = m2 | (m2>>16); // needed to go up to 32-bit integers
    m2 = m2 + 1;

    // in line below need to rename to  __clz() in CUDA; see
    // 1.10 in https://docs.nvidia.com/cuda/cuda-math-api/
    m3 = 1 << (32 - __builtin_clz(n-1));  //  needs n>1

    printf("n, m1, m2, m3 = %2d, %2d, %2d, %2d \n",n,m1,m2,m3);
  }
  
}

