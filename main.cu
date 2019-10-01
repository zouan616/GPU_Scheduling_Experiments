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
cudaStream_t stream;
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
    kernel_deadline[n];
    int CTA_num[n];

    for(int i = 0; i < n; i++)
    {
    GPU_task[i].task_num = i;
    GPU_task[i].ready = false;
    }


    //create CPU thread
    pthread_t tidp[n];
    struct para GPU_para[n];

    
    //create CUDA stream
    for(int i = 0; i < n; i++)
    {
    cudaStream_t stream[i];
    cudaStreamCreate(&stream[i]);
    }



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

    int CTA_num1_start, CTA_num1_end;
    int CTA_num2_start, CTA_num2_end;
    int CTA_num3_start, CTA_num3_end;
    int CTA_num4_start, CTA_num4_end;
    int CTA_num5_start, CTA_num5_end;

    CTA_num1_start = 0;
    CTA_num1_end = CTA_num1_start + CTA_num1 - 1;

    CTA_num2_start = CTA_num1_end + 1;
    CTA_num2_end = CTA_num2_start + CTA_num2 - 1;

    CTA_num3_start = CTA_num2_end + 1;
    CTA_num3_end = CTA_num3_start + CTA_num3 - 1;

    CTA_num4_start = CTA_num3_end + 1;
    CTA_num4_end = CTA_num4_start + CTA_num4 - 1;

    CTA_num5_start = CTA_num4_end + 1;
    CTA_num5_end = CTA_num5_start + CTA_num5 - 1;

    long int nBytes = N * sizeof(float);
    

    //Apply for host memory
    long int nBytes = N * sizeof(float);
    float *x[n], *y[n], *z[n];

    for(int i = 0; i < n; i++)
    {
    checkCuda(cudaMallocHost((void **) &x[n], nBytes));
    checkCuda(cudaMallocHost((void **) &y[n], nBytes));
    checkCuda(cudaMallocHost((void **) &z[n], nBytes));
        //Initialize data
        for(long int i = 0; i < N; i++)
        {
        x[i] = i % 20;
        y[i] = i % 20;
        z[i] = 0;
        }
        
    }


    //Apply for GPU memory
    float *d_x[n], *d_y[n], *d_z[n];

    for(int i = 0; i < n; i++)
    {
    checkCuda(cudaMalloc((void **) &d_x[n], nBytes));
    checkCuda(cudaMalloc((void **) &d_y[n], nBytes));
    checkCuda(cudaMalloc((void **) &d_z[n], nBytes));
        //Initialize data
        for(long int i = 0; i < N; i++)
        {
        x[i] = i % 20;
        y[i] = i % 20;
        z[i] = 0;
        }
        
    }

    //Initial grid size
    for(int i = 0; i < n; i++)
    {
    dim3 blocksize[i](1024);
    dim3 gridsize[i](56);   
    }
  


    para1.stream = stream1;
    para1.gridsize = gridsize1;
    para1.blocksize = blocksize1;
    para1.SM_num_start = CTA_num1_start;
    para1.SM_num_end = CTA_num1_end;
    para1.d_data01 = x1;
    para1.d_data02 = y1;
    para1.d_data03 = z1;
    para1.d_data1 = d_x1;
    para1.d_data2 = d_y1;
    para1.d_data3 = d_z1;
    para1.N = N;
    para1.memory_deadline = memory_deadline_1;
    para1.kernel_deadline = kernel_deadline_1;
    para1.nBytes = nBytes;

    para2.stream = stream2;
    para2.gridsize = gridsize2;
    para2.blocksize = blocksize2;
    para2.SM_num_start = CTA_num2_start;
    para2.SM_num_end = CTA_num2_end;
    para2.d_data01 = x2;
    para2.d_data02 = y2;
    para2.d_data03 = z2;
    para2.d_data1 = d_x2;
    para2.d_data2 = d_y2;
    para2.d_data3 = d_z2;
    para2.N = N;
    para2.memory_deadline = memory_deadline_2;
    para2.kernel_deadline = kernel_deadline_2;
    para2.nBytes = nBytes;

    para3.stream = stream3;
    para3.gridsize = gridsize3;
    para3.blocksize = blocksize3;
    para3.SM_num_start = CTA_num3_start;
    para3.SM_num_end = CTA_num3_end;
    para3.d_data01 = x3;
    para3.d_data02 = y3;
    para3.d_data03 = z3;
    para3.d_data1 = d_x3;
    para3.d_data2 = d_y3;
    para3.d_data3 = d_z3;
    para3.N = N;
    para3.memory_deadline = memory_deadline_3;
    para3.kernel_deadline = kernel_deadline_3;
    para3.nBytes = nBytes;


    para4.stream = stream4;
    para4.gridsize = gridsize4;
    para4.blocksize = blocksize4;
    para4.SM_num_start = CTA_num4_start;
    para4.SM_num_end = CTA_num4_end;
    para4.d_data01 = x4;
    para4.d_data02 = y4;
    para4.d_data03 = z4;
    para4.d_data1 = d_x4;
    para4.d_data2 = d_y4;
    para4.d_data3 = d_z4;
    para4.N = N;
    para4.memory_deadline = memory_deadline_4;
    para4.kernel_deadline = kernel_deadline_4;
    para4.nBytes = nBytes;

    para5.stream = stream5;
    para5.gridsize = gridsize5;
    para5.blocksize = blocksize5;
    para5.SM_num_start = CTA_num5_start;
    para5.SM_num_end = CTA_num5_end;
    para5.d_data01 = x5;
    para5.d_data02 = y5;
    para5.d_data03 = z5;
    para5.d_data1 = d_x5;
    para5.d_data2 = d_y5;
    para5.d_data3 = d_z5;
    para5.N = N;
    para5.memory_deadline = memory_deadline_5;
    para5.kernel_deadline = kernel_deadline_5;
    para5.nBytes = nBytes;
 
    struct timeval tv;
    gettimeofday(&tv,NULL);
    time_offset = tv.tv_sec*1000 + tv.tv_usec/1000;
      
      pthread_create(&tidp[0], NULL, pthread0, (void *)& para1);

      pthread_create(&tidp[1], NULL, pthread1, (void *)& para2);

      pthread_create(&tidp[2], NULL, pthread2, (void *)& para3);

      pthread_create(&tidp[3], NULL, pthread3, (void *)& para4);

      pthread_create(&tidp[4], NULL, pthread4, (void *)& para5);


    
    
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
    cudaFree(d_x1);
    cudaFree(d_y1);
    cudaFree(d_x2);
    cudaFree(d_y2);
    cudaFree(d_x3);
    cudaFree(d_y3);
    cudaFree(d_x4);
    cudaFree(d_y4);
    cudaFree(d_x5);
    cudaFree(d_y5);

    //Free host memory
    free(x1);
    free(y1);
    free(x2);
    free(y2);
    free(x3);
    free(y3);
    free(x4);
    free(y4);
    free(x5);
    free(y5);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
    cudaStreamDestroy(stream5);

    
    return 0;
}


void * pthread0(void *data)       
{
    // pin to a core
    cpu_set_t cpuset1;
    CPU_ZERO(&cpuset1);
    CPU_SET(1, &cpuset1);
    int s;
    s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset1);
    if (s != 0)
        cout << "Fail to pin to core 1" << endl;

    struct para* tt = (struct para*)data;

    struct timeval tv;
    double time_ms_start;
    double time_ms_end;
    double duration;

    gettimeofday(&tv,NULL);
    time_ms_start = tv.tv_sec*1000 + tv.tv_usec/1000 - time_offset;

    task1.absolute_deadline = time_ms_start + tt->memory_deadline;
    

    for(int i = 0; i < memory_length1; i++)
    {
    cudaMemcpyAsync((void*)tt->d_data03, (void*)tt->d_data3, tt->nBytes, cudaMemcpyDeviceToHost,tt->stream);
    cudaMemcpyAsync((void*)tt->d_data1, (void*)tt->d_data01, tt->nBytes, cudaMemcpyHostToDevice,tt->stream);
    cudaMemcpyAsync((void*)tt->d_data2, (void*)tt->d_data02, tt->nBytes, cudaMemcpyHostToDevice,tt->stream);
    }
    cudaStreamSynchronize(tt->stream);
    
    for(int i = 0; i < kernel_length1; i++)
    {
    // task_1 <<<tt->gridsize,tt->blocksize,0,tt->stream>>> (tt->d_data1, tt->d_data2, tt->d_data3, tt->N, tt->SM_num_start, tt->SM_num_end);
    }


    cudaStreamSynchronize(tt->stream);


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

void * pthread1(void *data)       
{
    // pin to a core
    cpu_set_t cpuset2;
    CPU_ZERO(&cpuset2);
    CPU_SET(2, &cpuset2);
    int s;
    s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset2);
    if (s != 0)
        cout << "Fail to pin to core 2" << endl;

   struct para* tt = (struct para*)data;
   struct timeval tv;
   double time_ms_start;
   double time_ms_end;
   double duration;

 
    gettimeofday(&tv,NULL);
    time_ms_start = tv.tv_sec*1000 + tv.tv_usec/1000 - time_offset;


    //cout << "Task 2 begin execution" << endl;
    for(int i = 0; i < memory_length2; i++)
    {
    cudaMemcpyAsync((void*)tt->d_data03, (void*)tt->d_data3, tt->nBytes, cudaMemcpyDeviceToHost,tt->stream);
    cudaMemcpyAsync((void*)tt->d_data1, (void*)tt->d_data01, tt->nBytes, cudaMemcpyHostToDevice,tt->stream);
    cudaMemcpyAsync((void*)tt->d_data2, (void*)tt->d_data02, tt->nBytes, cudaMemcpyHostToDevice,tt->stream);
    }
    cudaStreamSynchronize(tt->stream);

    
    for(int i = 0; i < kernel_length2; i++)
    {
    //task_2 <<<tt->gridsize,tt->blocksize,0,tt->stream>>> (tt->d_data1, tt->d_data2, tt->d_data3, tt->N, tt->SM_num_start, tt->SM_num_end);
    }

    cudaStreamSynchronize(tt->stream);

    gettimeofday(&tv,NULL);
    time_ms_end = tv.tv_sec*1000 + tv.tv_usec/1000 - time_offset;
    duration = time_ms_end - time_ms_start;
    if(duration > tt->kernel_deadline)
    {
    cout << "2_1 miss, duration:" << duration << endl;
    exit(0);
    }

    //cout << "2_1 pass" << endl;

    return 0;
}

void * pthread2(void *data)       
{
    // pin to a core
    cpu_set_t cpuset3;
    CPU_ZERO(&cpuset3);
    CPU_SET(3, &cpuset3);
    int s;
    s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset3);
    if (s != 0)
        cout << "Fail to pin to core 3" << endl;

  struct para* tt = (struct para*)data;

  struct timeval tv;
  double time_ms_start;
  double time_ms_end;
  double duration;

    gettimeofday(&tv,NULL);
    time_ms_start = tv.tv_sec*1000 + tv.tv_usec/1000 - time_offset;

    task3.absolute_deadline = time_ms_start + tt->memory_deadline;
    

    //cout << "Task 3 begin execution" << endl;
    for(int i = 0; i < memory_length3; i++)
    {
    cudaMemcpyAsync((void*)tt->d_data03, (void*)tt->d_data3, tt->nBytes, cudaMemcpyDeviceToHost,tt->stream);
    cudaMemcpyAsync((void*)tt->d_data1, (void*)tt->d_data01, tt->nBytes, cudaMemcpyHostToDevice,tt->stream);
    cudaMemcpyAsync((void*)tt->d_data2, (void*)tt->d_data02, tt->nBytes, cudaMemcpyHostToDevice,tt->stream);
    }
    cudaStreamSynchronize(tt->stream);


    for(int i = 0; i < kernel_length3; i++)
    {
    //task_3 <<<tt->gridsize,tt->blocksize,0,tt->stream>>> (tt->d_data1, tt->d_data2 , tt->d_data3, tt->N, tt->SM_num_start, tt->SM_num_end);
    }


    cudaStreamSynchronize(tt->stream);

    gettimeofday(&tv,NULL);
    time_ms_end = tv.tv_sec*1000 + tv.tv_usec/1000 - time_offset;

    duration = time_ms_end - time_ms_start;
    if(duration > tt->kernel_deadline)
    {
    cout << "3_1 miss, duration:" << duration << endl;
    exit(0);
    }

    //cout << "3_1 pass" << endl;

    return 0;
}

void * pthread3(void *data)       
{
    // pin to a core
    cpu_set_t cpuset4;
    CPU_ZERO(&cpuset4);
    CPU_SET(4, &cpuset4);
    int s;
    s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset4);
    if (s != 0)
        cout << "Fail to pin to core 4" << endl;


  struct para* tt = (struct para*)data;
  struct timeval tv;
  double time_ms_start;
  double time_ms_end;
  double duration;


    gettimeofday(&tv,NULL);
    time_ms_start = tv.tv_sec*1000 + tv.tv_usec/1000 - time_offset;

    task4.absolute_deadline = time_ms_start + tt->memory_deadline;
    

    //cout << "Task 4 begin execution" << endl;
    for(int i = 0; i < memory_length4; i++)
    {
    cudaMemcpyAsync((void*)tt->d_data03, (void*)tt->d_data3, tt->nBytes, cudaMemcpyDeviceToHost,tt->stream);
    cudaMemcpyAsync((void*)tt->d_data1, (void*)tt->d_data01, tt->nBytes, cudaMemcpyHostToDevice,tt->stream);
    cudaMemcpyAsync((void*)tt->d_data2, (void*)tt->d_data02, tt->nBytes, cudaMemcpyHostToDevice,tt->stream);
    }
    cudaStreamSynchronize(tt->stream);


    for(int i = 0; i < kernel_length4; i++)
    {
    //task_4 <<<tt->gridsize,tt->blocksize,0,tt->stream>>> (tt->d_data1, tt->d_data2, tt->d_data3 , tt->N, tt->SM_num_start, tt->SM_num_end); 
    }


    cudaStreamSynchronize(tt->stream);

    gettimeofday(&tv,NULL);
    time_ms_end = tv.tv_sec*1000 + tv.tv_usec/1000 - time_offset;

    duration = time_ms_end - time_ms_start;
    if(duration > tt->kernel_deadline)
    {
    cout << "4_1 miss, duration:" << duration << endl;
    exit(0);
    }

    //cout << "4_1 pass" << endl;

    return 0;
}


void * pthread4(void *data)       
{
    // pin to a core
    cpu_set_t cpuset5;
    CPU_ZERO(&cpuset5);
    CPU_SET(5, &cpuset5);
    int s;
    s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset5);
    if (s != 0)
        cout << "Fail to pin to core 5" << endl;

  

  struct para* tt = (struct para*)data;

  struct timeval tv;
  double time_ms_start;
  double time_ms_end;
  double duration;

 
    gettimeofday(&tv,NULL);
    time_ms_start = tv.tv_sec*1000 + tv.tv_usec/1000 - time_offset;

    task5.absolute_deadline = time_ms_start + tt->memory_deadline;

    //cout << "Task 5 begin execution" << endl;
    for(int i = 0; i < memory_length5; i++)
    {
    cudaMemcpyAsync((void*)tt->d_data03, (void*)tt->d_data3, tt->nBytes, cudaMemcpyDeviceToHost,tt->stream);
    cudaMemcpyAsync((void*)tt->d_data1, (void*)tt->d_data01, tt->nBytes, cudaMemcpyHostToDevice,tt->stream);
    cudaMemcpyAsync((void*)tt->d_data2, (void*)tt->d_data02, tt->nBytes, cudaMemcpyHostToDevice,tt->stream);
    }
    cudaStreamSynchronize(tt->stream);
    
    for(int i = 0; i < kernel_length5; i++)
    {
    //task_5 <<<tt->gridsize,tt->blocksize,0,tt->stream>>> (tt->d_data1, tt->d_data2, tt->d_data3, tt->N, tt->SM_num_start, tt->SM_num_end);
    }

    cudaStreamSynchronize(tt->stream);

    gettimeofday(&tv,NULL);
    time_ms_end = tv.tv_sec*1000 + tv.tv_usec/1000 - time_offset;

    duration = time_ms_end - time_ms_start;
    if(duration > tt->kernel_deadline)
    {
    cout << "5_1 miss, duration:" << duration << endl;
    exit(0);
    }

    return 0;
}
