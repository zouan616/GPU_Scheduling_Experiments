#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "device_launch_parameters.h"
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
#include <pthread.h>
#include <errno.h>
#include <vector>
#include <assert.h>


using namespace std;



struct para
{
int task_num;
int iter;
dim3 gridsize;
dim3 blocksize;
int SM_num_start;
int SM_num_end;
float *d_data01;
float *d_data02;
float *d_data03;
float *d_data1;
float *d_data2;
float *d_data3;
float *d_result;
long int N;
float memory_deadline;
float kernel_deadline;
long int nBytes;
};

struct task
{
int task_num;
bool ready;
float current_time;
float absolute_deadline;
float priority;
};

double time_offset;
double global_time;

int memory_length1;
int memory_length2;
int memory_length3;
int memory_length4;
int memory_length5;

int kernel_length1;
int kernel_length2;
int kernel_length3;
int kernel_length4;
int kernel_length5;


#include "task.cu"


inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}


//main function
int main(int argc,char *argv[])
{
    cpu_set_t cpuset0;
    CPU_ZERO(&cpuset0);
    CPU_SET(0, &cpuset0);
    int s;
    s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset0);
    if (s != 0)
        cout << "Fail to pin to core 0" << endl;

    //Initialize task queque


    //task number
    int n = 5; 

    struct task GPU_task[n];
    int memory_length[n];
    float memory_deadline[n];
    int kernel_length[n];
    float kernel_deadline[n];
    int CTA_num[n];

    for(int i = 0; i < n; i++)
    {
    GPU_task[i].task_num = i;
    GPU_task[i].ready = false;
    }


    //create CPU thread
    pthread_t tidp[n];
    struct para GPU_para[n];



    //data length
    long int N = 1 << 18;



    ifstream config_input_parameter(argv[1],ios::app);
    

    cout << argv[1] << endl;

    for(int i = 0; i < n; i++)
    {
    config_input_parameter >> memory_length[n];
    config_input_parameter >> memory_deadline[n];
    config_input_parameter >> kernel_length[n];
    config_input_parameter >> kernel_deadline[n];
    }
    




    config_input_parameter.close();

    int CTA_num_start[n], CTA_num_end[n];

    for(int i = 0; i < n; i++)
    {
    if(i==0)
        {
        CTA_num_start[i] = 0;
        CTA_num_end[i] = CTA_num_start[i] + CTA_num[i] - 1;
        }
    CTA_num_start[i] = CTA_num_end[i-1] + 1;
    CTA_num_end[i] = CTA_num_start[i] + CTA_num[i] - 1;
    }


    
     long int nBytes = N * sizeof(float);

    //Apply for host memory
    
    float *x[n], *y[n], *z[n];

    for(int i = 0; i < n; i++)
    {
    checkCuda(cudaMallocHost((void **) &x[n], nBytes));
    checkCuda(cudaMallocHost((void **) &y[n], nBytes));
    checkCuda(cudaMallocHost((void **) &z[n], nBytes));
        //Initialize data
        for(int j = 0; j < N; j++)
        {
        x[i][j] = j % 20;
        y[i][j] = j % 20;
        z[i][j] = 0;
        }
        
    }


    //Apply for GPU memory
    float *d_x[n], *d_y[n], *d_z[n];

    for(int i = 0; i < n; i++)
    {
    checkCuda(cudaMalloc((void **) &d_x[n], nBytes));
    checkCuda(cudaMalloc((void **) &d_y[n], nBytes));
    checkCuda(cudaMalloc((void **) &d_z[n], nBytes));       
    }

    //Initial grid size

    dim3 blocksize(1024);
    dim3 gridsize(56);   

    //passing data
    struct para para[n];
    for(int i = 0; i < n; i++)
    {
    para[i].task_num = i;
    para[i].gridsize = gridsize;
    para[i].blocksize = blocksize;
    para[i].SM_num_start = CTA_num_start[i];
    para[i].SM_num_end = CTA_num_end[i];
    para[i].d_data01 = x[i];
    para[i].d_data02 = y[i];
    para[i].d_data03 = z[i];
    para[i].d_data1 = d_x[i];
    para[i].d_data2 = d_y[i];
    para[i].d_data3 = d_z[i];
    para[i].N = N;
    para[i].memory_deadline = memory_deadline[i];
    para[i].kernel_deadline = kernel_deadline[i];
    para[i].nBytes = nBytes;
    }

    

 
    struct timeval tv;
    gettimeofday(&tv,NULL);
    time_offset = tv.tv_sec*1000 + tv.tv_usec/1000;
      

    for(int i = 0; i < n; i++)
    {
      pthread_create(&tidp[0], NULL, pthread0, (void *)& para[i]);
    }

    
    
    //End to launch threads in First deadline first 

    
    //cout << "got here2" << endl;

    usleep(200000000);
    
    
    //pthread_cancel(tidp[0]);
    //pthread_cancel(tidp[1]);
    //pthread_cancel(tidp[2]);
    //pthread_cancel(tidp[3]);
    //pthread_cancel(tidp[4]);

    //pthread_join(tidp[0],NULL);
    //pthread_join(tidp[1],NULL);
    //pthread_join(tidp[2],NULL);
    //pthread_join(tidp[3],NULL);
    //pthread_join(tidp[4],NULL);
    
    

    //cout << "got here3" << endl;

    //cudaMemcpy((void*)z1, (void*)d_z1, nBytes, cudaMemcpyDeviceToHost);
    //cudaMemcpy((void*)z2, (void*)d_z2, nBytes, cudaMemcpyDeviceToHost);
    //cudaMemcpy((void*)z3, (void*)d_z3, nBytes, cudaMemcpyDeviceToHost);
    //cudaMemcpy((void*)z4, (void*)d_z4, nBytes, cudaMemcpyDeviceToHost);
    //cudaMemcpy((void*)z5, (void*)d_z5, nBytes, cudaMemcpyDeviceToHost);

    //Free device memory
    for(int i = 0; i < n; i++)
    {
    cudaFree(d_x[i]);
    cudaFree(d_y[i]);
    cudaFree(d_z[i]);      
    }


    //Free host memory
    for(int i = 0; i < n; i++)
    {
    free(x[i]);
    free(y[i]);
    free(z[i]);  
    }

    
    return 0;
}


void * pthread0(void *data)       
{

    struct para* tt = (struct para*)data;

    // pin to a core
    cpu_set_t cpuset1;
    CPU_ZERO(&cpuset1);
    CPU_SET(1+tt->task_num, &cpuset1);
    int s;
    s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset1);
    if (s != 0)
        cout << "Fail to pin to core " << 1+tt->task_num << endl;


    cudaStream_t stream;
    cudaStreamCreate(&stream);


    struct timeval tv;
    double time_ms_start;
    double time_ms_end;
    double duration;

    gettimeofday(&tv,NULL);
    time_ms_start = tv.tv_sec*1000 + tv.tv_usec/1000 - time_offset;

    task1.absolute_deadline = time_ms_start + tt->memory_deadline;
    

    for(int i = 0; i < memory_length1; i++)
    {
    cudaMemcpyAsync((void*)tt->d_data03, (void*)tt->d_data3, tt->nBytes, cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync((void*)tt->d_data1, (void*)tt->d_data01, tt->nBytes, cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync((void*)tt->d_data2, (void*)tt->d_data02, tt->nBytes, cudaMemcpyHostToDevice,stream);
    }
    cudaStreamSynchronize(stream);
    
    for(int i = 0; i < kernel_length1; i++)
    {
    task <<<tt->gridsize,tt->blocksize,0,stream>>> (tt->d_data1, tt->d_data2, tt->d_data3, tt->N, tt->SM_num_start, tt->SM_num_end);
    }


    cudaStreamSynchronize(stream);


    gettimeofday(&tv,NULL);
    time_ms_end = tv.tv_sec*1000 + tv.tv_usec/1000 - time_offset;
    
    duration = time_ms_end - time_ms_start;

    if(duration > tt->kernel_deadline)
    {
    cout << "1_1 miss, duration:" << duration << endl;
    exit(0);
    }

    //cout << "1_1 pass" << endl;

    return 0;
}

