#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wave_2d.h"
#include <time.h>
#define BLOCK_NUM 32
#define THREAD_NUM 512


__global__ void kernel_cuda_update(double *olddata, double *data, double *newdata, double C, double K, double dt){
	
	const int tid = threadIdx.x;
	const int bid = blockIdx.x; 
	//if(tid == 0)printf("len = %d\n", len);
	int x, i, j;
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
		//printf("X = %d, new = ")
	}


}
__global__ void cuda_move_data(double *olddata, double *data, double *newdata){
	const int tid = threadIdx.x;
	const int bid = blockIdx.x; 
	int i;
	for(i = tid + bid*THREAD_NUM; i < ARR_SZ; i += BLOCK_NUM*THREAD_NUM){
		olddata[i] = data[i];
		data[i] = newdata[i];
	}

}
void cuda_update(double* olddata, double* data, double* newdata,double C,double K, double dt){
	double *gpu_data, *gpu_old, *gpu_new;
	cudaMalloc((void**) &gpu_data, sizeof(double)*ARR_SZ);
	cudaMalloc((void**) &gpu_old, sizeof(double)*ARR_SZ);
	cudaMalloc((void**) &gpu_new, sizeof(double)*ARR_SZ);
	cudaMemcpy(gpu_data, data, sizeof(double)*ARR_SZ, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_old, olddata, sizeof(double)*ARR_SZ, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_new, newdata, sizeof(double)*ARR_SZ, cudaMemcpyHostToDevice);
	//int _num = ARR_SZ/THREAD_NUM + 1;
	//if(_num > BLOCK_NUM) _num = BLOCK_NUM;
	int i;
	for(i = 1;i <= 2000; ++i){
		kernel_cuda_update<<< BLOCK_NUM, THREAD_NUM>>>(gpu_old, gpu_data, gpu_new,C, K, dt);
		cuda_move_data<<<BLOCK_NUM, THREAD_NUM>>>(gpu_old, gpu_data, gpu_new);
	}	
	cudaMemcpy(olddata, gpu_old, sizeof(double)*ARR_SZ, cudaMemcpyDeviceToHost);
	cudaMemcpy(data, gpu_data, sizeof(double)*ARR_SZ, cudaMemcpyDeviceToHost);
	cudaMemcpy(newdata, gpu_new, sizeof(double)*ARR_SZ, cudaMemcpyDeviceToHost);
	cudaFree(gpu_data);
	cudaFree(gpu_old);
	cudaFree(gpu_new);
	
		}
int main(){
    int i, j;
    double dt = 0.04, C = 16, K = 0.1, h = 6;
    double *data, *olddata, *newdata;//*tmp;
    double x[PEAK_SZ][PEAK_SZ], linspace[PEAK_SZ], delta = 2.0/(PEAK_SZ-1.0);
    data = (double*)malloc(sizeof(double)*ARR_SZ);
    olddata = (double*)malloc(sizeof(double)*ARR_SZ);
    newdata = (double*)malloc(sizeof(double)*ARR_SZ);
    for(i = 0; i < ARR_SZ; i++){
        data[i] = 1.0;
    }

    for(i = 0; i < PEAK_SZ; i++){
        linspace[i] = -1.0 + delta * i;
    }

    for(i = 0; i < PEAK_SZ; i++){
        for(j = 0; j < PEAK_SZ; j++){
            x[i][j] = linspace[i];
        }
    }
    for(i = 0; i < PEAK_SZ; i++){
        for(j = 0; j < PEAK_SZ; j++){
            data[(i+20)*GRID_SZ+j+20] += h * exp( -5 * (pow(x[i][j], 2 ) + pow(x[j][i], 2 )));
        }
    }
    for(i = 0; i < ARR_SZ; i++){
        olddata[i] = data[i];
    }

    /*for(i = 0; i < 2000; i++){
        sequential_update( data, olddata, newdata, C, K, dt);
        tmp = olddata;
        olddata = data;
        data = newdata;
        newdata = tmp;
    }*/
	clock_t start = clock();
	
	cuda_update(olddata, data, newdata, C, K, dt);
    
	double use = clock() - start;
	
	
	for(i = 0; i < GRID_SZ; i++){
        for(j = 0; j < GRID_SZ; j++){
            printf("%lf, ", data[i*GRID_SZ+j]);
        }
        printf("\n");
    }

	printf("time use: %lf\n", use/CLOCKS_PER_SEC);
}
void sequential_update(double *data, double *olddata, double *newdata, double C, double K, double dt ){
    int i, j, add_i, sub_i, add_j, sub_j;
    double pot;
    for( i = 0; i < GRID_SZ; i++){
        for( j = 0; j < GRID_SZ; j++){
            add_i = i+1 >= GRID_SZ ? i : i+1;
            add_j = j+1 >= GRID_SZ ? j : j+1;
            sub_i = i-1 < 0 ? 0 : i-1;
            sub_j = j-1 < 0 ? 0 : j-1;
            pot = data[add_i*GRID_SZ+j]+
                  data[sub_i*GRID_SZ+j]+
                  data[add_j+i*GRID_SZ]+
                  data[sub_j+i*GRID_SZ]-
                  4*data[i*GRID_SZ+j];
            newdata[i * GRID_SZ + j] = ( pow(C * dt, 2) * pot * 2 + 4 * data[i * GRID_SZ + j] - olddata[i * GRID_SZ + j] *(2 - K * dt) ) / (2 + K * dt);
        }
    }
}
