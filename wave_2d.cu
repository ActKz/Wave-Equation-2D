#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wave_2d.h"
#include <time.h>
#define BLOCK_NUM 32
#define THREAD_NUM 512

extern "C"{

__global__ void kernel_cuda_update(double *olddata, double *data, double *newdata, double C, double K, double dt, int step){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int x, i, j, y;
    int add_i, add_j, sub_i, sub_j;
        for(x = tid + bid*THREAD_NUM; x < ARR_SZ; x += THREAD_NUM*BLOCK_NUM){
        	i = x / GRID_SZ;
        	j = x % GRID_SZ;
            add_i = i+1 >= GRID_SZ ? i : i+1;
            add_j = j+1 >= GRID_SZ ? j : j+1;
        	sub_i = i-1 < 0 ? 0 : i - 1;
        	sub_j = j-1 < 0 ? 0 : j - 1;
        	double pot = data[add_i * GRID_SZ + j] +
        				 data[sub_i * GRID_SZ + j] +
        				 data[add_j + i * GRID_SZ] +
        				 data[sub_j + i * GRID_SZ] -
        				 4 * data[i * GRID_SZ + j] ;
        	double tmp = C * dt;
        	newdata[x] = ( tmp*tmp * pot * 2 + 4 * data[x] - olddata[x] *(2 - K * dt)) / (2 + K*dt);
        }
}

void cuda_update(double* olddata, double* data, double* newdata,double C,double K, double dt, int step){
    double *gpu_data, *gpu_old, *gpu_new, *tmp;
    cudaMalloc((void**) &gpu_data, sizeof(double)*ARR_SZ);
    cudaMalloc((void**) &gpu_old, sizeof(double)*ARR_SZ);
    cudaMalloc((void**) &gpu_new, sizeof(double)*ARR_SZ);
    cudaMemcpy(gpu_data, data, sizeof(double)*ARR_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_old, olddata, sizeof(double)*ARR_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_new, newdata, sizeof(double)*ARR_SZ, cudaMemcpyHostToDevice);
    int i;
    for(i = 1;i <= step; ++i){
    	kernel_cuda_update<<< BLOCK_NUM, THREAD_NUM>>>(gpu_old, gpu_data, gpu_new,C, K, dt, step);
        tmp = gpu_old;
        gpu_old = gpu_data;
        gpu_data = gpu_new;
        gpu_new = tmp;
    }
    cudaMemcpy(data, gpu_data, sizeof(double)*ARR_SZ, cudaMemcpyDeviceToHost);
    cudaFree(gpu_data);
    cudaFree(gpu_old);
    cudaFree(gpu_new);
}

}
