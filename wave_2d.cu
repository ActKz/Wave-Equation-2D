#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wave_2d.h"
#include <time.h>

#define THREAD_NUM 512
#define BLOCK_NUM 32


__global__ void  kernel_cuda_update(float *olddata, float *data, float *newdata,
                                    int SIZE, int steps, int grid_sz, float C, float K, float dt){
    //printf("hi\n");
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const int len = gridDim.x*blockDim.x;
    int i, k;
    int add_x, sub_x, x;
    int add_y, sub_y, y;
    
	for(i = tid; i < SIZE; i+= len){
		//if(k == 1) printf("%f\n",data[i]);
		x = i / grid_sz;
		y = i % grid_sz;
		add_x = x+1 >= grid_sz ? x: x+1;
		sub_x = x-1 < 0 ? 0: x-1;
		add_y = y+1 >= grid_sz ? y: y+1;
		sub_y = y-1 < 0 ? 0: y+1;
		float pot =  data[add_x*grid_sz+y]+
					  data[sub_x*grid_sz+y]+
					  data[add_y+x*grid_sz]+
					  data[sub_y+x*grid_sz]-
					  4*data[x*grid_sz+y];
		float opr = C * dt;
		newdata[i] = ( opr*opr * pot * 2 + 4 * data[i] - olddata[i] *(2 - K * dt) ) / (2 + K * dt);
		
		//printf("QQ\n");
	}   
	__syncthreads();
	for(k = tid; k < SIZE; k+= len){
		olddata[k] = data[k];
		data[k] = newdata[k];
		//printf("%f\n", data[k]);
	}
   __syncthreads();
   /*if(tid == 0 && bid == 0){
		for(i = 0;i < GRID_SZ; ++i){
			for(k = 0;k < GRID_SZ; ++k){
				printf("%4.3lf, ",data[i*GRID_SZ+k]);
			}
			printf("\n");
		}
   }*/
}
int main(){
    int i, j;
    float dt = 0.04, C = 16, K = 0.1, h = 6;
    float *data, *olddata, *newdata; //*tmp;
    float x[PEAK_SZ][PEAK_SZ], linspace[PEAK_SZ], delta = 2.0/(PEAK_SZ-1.0);
    data = (float*)malloc(sizeof(float)*ARR_SZ);
    olddata = (float*)malloc(sizeof(float)*ARR_SZ);
    newdata = (float*)malloc(sizeof(float)*ARR_SZ);
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
    /*for(i = 0; i < 20; i++){
        sequential_update( data, olddata, newdata, C, K, dt);
        tmp = olddata;
        olddata = data;
        data = newdata;
        newdata = tmp;
    }*/
    clock_t start = clock();
    cuda_update(data, olddata, newdata, C, K, dt);

   for(i = 0; i < GRID_SZ; i++){
        for(j = 0; j < GRID_SZ; j++){
            printf("%f, ", data[i*GRID_SZ+j]);
        }
        printf("\n");
    }
    float T = clock() - start;
    printf("time use : %lf\n", T/CLOCKS_PER_SEC);

}
void sequential_update(float *data, float *olddata, float *newdata, float C, float K, float dt ){
    int i, j, add_i, sub_i, add_j, sub_j;
    float pot;
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
void cuda_update(float *data, float *olddata, float *newdata, float C, float K, float dt ){
    float *gpu_old, *gpu_val, *gpu_new;
    
    cudaMalloc((void**) &gpu_old, sizeof(float)*ARR_SZ);
    cudaMalloc((void**) &gpu_val, sizeof(float)*ARR_SZ);
    cudaMalloc((void**) &gpu_new, sizeof(float)*ARR_SZ);
    
    cudaMemcpy(gpu_old, olddata,sizeof(float)*ARR_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_val, data,sizeof(float)*ARR_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_new, newdata,sizeof(float)*ARR_SZ, cudaMemcpyHostToDevice);
    printf("hello\n");
    
    //for(i = 1;i <= nsteps; ++i){
   // int _num = ARR_SZ/THREAD_NUM;
    //if(_num > BLOCK_NUM) _num = BLOCK_NUM;
	int i;
	for(i = 1;i <= STEP; ++i){
		kernel_cuda_update<<< BLOCK_NUM, THREAD_NUM>>>( gpu_old, gpu_val, gpu_new, 
                                        ARR_SZ, STEP,GRID_SZ,
                                        C, K, dt);
	}
   
   // }
    cudaMemcpy(olddata, gpu_old,sizeof(float)*ARR_SZ, cudaMemcpyDeviceToHost);
    cudaMemcpy(data, gpu_val,sizeof(float)*ARR_SZ, cudaMemcpyDeviceToHost);
    cudaMemcpy(newdata, gpu_new,sizeof(float)*ARR_SZ, cudaMemcpyDeviceToHost);
    cudaFree(gpu_new);
    cudaFree(gpu_old);
    cudaFree(gpu_val);



}
