#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "device_launch_parameters.h"
#include <stdint.h>
#include <iostream>
#include <stdlib.h>      
#include <time.h>  
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <sys/time.h>
#include <fstream>
#include "timespec_functions.h"
#include <algorithm>

using namespace std;

//global function
__global__ void task(float* x, float * y, float* z, int n, int SM_num_start, int SM_num_end){

int SM_num;
SM_num = __mysmid();


if((SM_num_start <= SM_num)&&(SM_num <= SM_num_end))
{    
    // Key technique use the (__mysmid() - SM_num_start) as blockIdx.x
    // global index
    long int index = threadIdx.x + (SM_num - SM_num_start) * blockDim.x;
    // step
    // interleaved execution
    long int off_set = blockDim.x * (SM_num_end - SM_num_start + 1);
    
    if(blockIdx.x < 28)
    {
        for (long int i = index; i < n/2; i += off_set)
        {
            z[i] = x[i] + y[i];
            for(int j = 0; j< 100; j++)
            {
            z[i] = z[i] + x[i] + y[i];
            }
        }
    }
    else
    {
        for (long int i = index + n/2; i < n; i += off_set)
        {
            z[i] = x[i] + y[i];
            for(int j = 0; j< 100; j++)
            {
            z[i] = z[i] + x[i] + y[i];
            }
        }
    }
//printf("SM num: %d ; Block id: %d \n", SM_num, blockIdx.x);
}

}
