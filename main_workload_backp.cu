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
void * pthread0(void *data);
void * scheduler(void *data);

struct para
{
int task_num;
int iter;
int memory_length;
int kernel_length;
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
long int nBytes;
};

struct task
{
int task_num;
bool ready;
bool memory_finish;
bool kernel_finish;
};




//task number
int n = 8; 
struct task GPU_task[8];

struct timeval global_tv[8];
double global_start_time[8];
double global_memory_start_time[8];
double global_memory_finish_time[8];
double global_kernel_start_time[8];
double global_kernel_finish_time[8];
double global_duration[8];

// Scheduler order


int sched_order[] = {1};






struct timeval offset_tv;
double offset_start_time;


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


    int memory_length[n];
    int kernel_length[n];
    int CTA_num[n];

    for(int i = 0; i < n; i++)
    {
    GPU_task[i].task_num = i;
    GPU_task[i].ready = false;
    GPU_task[i].memory_finish = false;
    GPU_task[i].kernel_finish = false;
    }


    //create CPU thread
    pthread_t tidp[n];
    pthread_t scheduler_thread;



    //data length
    long int N = 1 << 18;



    ifstream config_input_parameter(argv[1],ios::app);
    

    //cout << argv[1] << endl;

    for(int i = 0; i < n; i++)
    {
    config_input_parameter >> memory_length[i];
    config_input_parameter >> kernel_length[i];
    config_input_parameter >> CTA_num[i];
    }
    



    config_input_parameter.close();

    int CTA_num_start[n], CTA_num_end[n];



     long int nBytes = N * sizeof(float);


    //Apply for host memory
    
    float *x[n], *y[n], *z[n];

    for(int i = 0; i < n; i++)
    {
    checkCuda(cudaMallocHost((void **) &x[i], nBytes));
    checkCuda(cudaMallocHost((void **) &y[i], nBytes));
    checkCuda(cudaMallocHost((void **) &z[i], nBytes));
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
    checkCuda(cudaMalloc((void **) &d_x[i], nBytes));
    checkCuda(cudaMalloc((void **) &d_y[i], nBytes));
    checkCuda(cudaMalloc((void **) &d_z[i], nBytes));       
    }

    //Initial grid size

    dim3 blocksize(1024);
    dim3 gridsize(56);   

    //passing data
    struct para GPU_para[n];
    for(int i = 0; i < n; i++)
    {
    GPU_para[i].task_num = i;
    GPU_para[i].gridsize = gridsize;
    GPU_para[i].blocksize = blocksize;
    GPU_para[i].memory_length = memory_length[i];
    GPU_para[i].kernel_length = kernel_length[i];
    GPU_para[i].SM_num_start = CTA_num_start[i];
    GPU_para[i].SM_num_end = CTA_num_end[i];
    GPU_para[i].d_data01 = x[i];
    GPU_para[i].d_data02 = y[i];
    GPU_para[i].d_data03 = z[i];
    GPU_para[i].d_data1 = d_x[i];
    GPU_para[i].d_data2 = d_y[i];
    GPU_para[i].d_data3 = d_z[i];
    GPU_para[i].N = N;
    GPU_para[i].nBytes = nBytes;
    }

    

 

      

    for(int i = 0; i < n; i++)
    {
      pthread_create(&tidp[i], NULL, pthread0, (void *)& GPU_para[i]);
    }


    usleep(1000000);

    pthread_create(&scheduler_thread, NULL, scheduler, NULL);

    
    gettimeofday(&offset_tv,NULL);
    offset_start_time = offset_tv.tv_sec*1000 + offset_tv.tv_usec/1000;
    
    
    for(int i = 0; i < n; i++)
    {
    pthread_join(tidp[i],NULL);    
    }


    for(int i = 0; i < n; i++)
    {
    //cout << "-----------------------------" << endl;
    //cout << "task " << i+1 << " memory start time: " << global_memory_start_time[i] << endl;
    //cout << "task " << i+1 << " memory finish time: " << global_memory_finish_time[i] << endl;
    //cout << "task " << i+1 << " kernel start time: " << global_kernel_start_time[i] << endl;
    //cout << "task " << i+1 << " kernel finish time: " << global_kernel_finish_time[i] << endl;

    cout << global_kernel_finish_time[i] - global_kernel_start_time[i] << endl;
    }


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
    cudaFreeHost(x[i]);
    cudaFreeHost(y[i]);
    cudaFreeHost(z[i]); 
    }

    //cout << "Finish!" << endl;
    return 0;
}



void * scheduler(void *data)       
{

    

    // pin to a core
    cpu_set_t cpuset1;
    CPU_ZERO(&cpuset1);
    CPU_SET(1, &cpuset1);
    int s;
    s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset1);
    if (s != 0)
        cout << "Fail to pin to core " << "scheduler" << endl;

    

    for(int i = 0; i < n; i++)
    {
    GPU_task[sched_order[i]-1].ready = true;
        while(GPU_task[sched_order[i]-1].ready == true)
        {
        
        }
    }


    return 0;
}

void * pthread0(void *data)       
{

    struct para* tt = (struct para*)data;

    // pin to a core
    cpu_set_t cpuset1;
    CPU_ZERO(&cpuset1);
    CPU_SET(2+tt->task_num, &cpuset1);
    int s;
    s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset1);
    if (s != 0)
        cout << "Fail to pin to core " << 2+tt->task_num << endl;

  

    cudaStream_t stream;
    cudaStreamCreate(&stream);


    while(GPU_task[tt->task_num].ready == false)
    {
    }

    
    //cout << "memory length: " << tt->memory_length << endl;

    gettimeofday(&global_tv[tt->task_num],NULL);
    global_memory_start_time[tt->task_num] = global_tv[tt->task_num].tv_sec*1000 + global_tv[tt->task_num].tv_usec/1000 - offset_start_time;

    for(int i = 0; i < tt->memory_length*285; i++)
    {
    cudaMemcpyAsync((void*)tt->d_data1, (void*)tt->d_data01, tt->nBytes, cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync((void*)tt->d_data2, (void*)tt->d_data02, tt->nBytes, cudaMemcpyHostToDevice,stream);
    //cudaMemcpyAsync((void*)tt->d_data03, (void*)tt->d_data3, tt->nBytes, cudaMemcpyDeviceToHost,stream);
    }
    cudaStreamSynchronize(stream);

    GPU_task[tt->task_num].ready = false;



    gettimeofday(&global_tv[tt->task_num],NULL);
    global_memory_finish_time[tt->task_num] = global_tv[tt->task_num].tv_sec*1000 + global_tv[tt->task_num].tv_usec/1000 - offset_start_time;

    GPU_task[tt->task_num].memory_finish = true;


    //---------alg-II--------------------//
    /*
    if((tt->task_num != 0)&&(tt->task_num != 1))
    {
    while((GPU_task[7].memory_finish == false))
    {

    }
    }
    */


    //---------alg-B--------------------//
    ///*
    while((GPU_task[0].memory_finish == false)||(GPU_task[1].memory_finish == false)||(GPU_task[2].memory_finish == false)||(GPU_task[3].memory_finish == false)||(GPU_task[4].memory_finish == false)||(GPU_task[5].memory_finish == false)||(GPU_task[6].memory_finish == false)||(GPU_task[7].memory_finish == false))
    {

    }

    if((tt->task_num == 0)||(tt->task_num == 1))
    {
    while((GPU_task[2].kernel_finish == false)||(GPU_task[3].kernel_finish == false)||(GPU_task[4].kernel_finish == false)||(GPU_task[5].kernel_finish == false)||(GPU_task[6].kernel_finish == false)||(GPU_task[7].kernel_finish == false))
    {

    }
    }
    //*/

    gettimeofday(&global_tv[tt->task_num],NULL);
    global_kernel_start_time[tt->task_num] = global_tv[tt->task_num].tv_sec*1000 + global_tv[tt->task_num].tv_usec/1000 - offset_start_time;

    task <<<tt->gridsize,tt->blocksize,0,stream>>> (tt->d_data1, tt->d_data2, tt->d_data3, tt->N, tt->SM_num_start, tt->SM_num_end, tt->kernel_length);


    cudaStreamSynchronize(stream);

    gettimeofday(&global_tv[tt->task_num],NULL);
    global_kernel_finish_time[tt->task_num] = global_tv[tt->task_num].tv_sec*1000 + global_tv[tt->task_num].tv_usec/1000 - offset_start_time;

    GPU_task[tt->task_num].kernel_finish = true;

 
    cudaStreamDestroy(stream);

    return 0;
}

