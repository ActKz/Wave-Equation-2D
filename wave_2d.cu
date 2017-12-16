#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wave_2d.h"



__global__ void  kernel_cuda_update(double *olddata, double *data, double *newdata,
                                    int SIZE, int steps, int grid_sz, double C, double K, double dt){

    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const int len = gridDim.x*blockDim.x;

    int i, k;
    int add_x, sub_x, x;
    int add_y, sub_y, y;
    
    for(k = 1; k <= steps; ++k){
        for(i = tid; i < SIZE; i+= len){
            x = i / grid_sz;
            y = i % grid_sz;
            add_x = x+1 >= grid_sz ? x: x+1;
            sub_x = x-1 < 0 ? 0: x-1;
            add_y = y+1 >= grid_sz ? y: y+1;
            sub_y = y-1 < 0 ? 0: y+1;
            double pot =  data[add_x*grid_sz+y]+
                          data[sub_x*grid_sz+y]+
                          data[add_y+x*grid_sz]+
                          data[sub_y+x*grid_sz]-
                          4*data[x*grid_sz+y];
            double opr = C * dt;
            newdata[i] = ( (opr*opr) * pot * 2 + 4 * data[i] - olddata[i] *(2 - K * dt) ) / (2 + K * dt);
            olddata[i] = data[i];
            data[i] = newdata[i];
        }   
    }
}
int main(){
    int i, j;
    double dt = 0.04, C = 16, K = 0.1, h = 6;
    double *data, *olddata, *newdata; //*tmp;
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

    /*for(i = 0; i < 20; i++){
        sequential_update( data, olddata, newdata, C, K, dt);
        tmp = olddata;
        olddata = data;
        data = newdata;
        newdata = tmp;
    }*/
    cuda_update(data, olddata, newdata, C, K, dt);

    for(i = 0; i < GRID_SZ; i++){
        for(j = 0; j < GRID_SZ; j++){
            printf("%lf, ", data[i*GRID_SZ+j]);
        }
        printf("\n");
    }

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
void cuda_update(double *data, double *olddata, double *newdata, double C, double K, double dt ){
    double *gpu_old, *gpu_val, *gpu_new;
    
    cudaMalloc((void**) &gpu_old, sizeof(double)*ARR_SZ);
    cudaMalloc((void**) &gpu_val, sizeof(double)*ARR_SZ);
    cudaMalloc((void**) &gpu_new, sizeof(double)*ARR_SZ);
    
    cudaMemcpy(gpu_old, olddata,sizeof(double)*ARR_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_val, data,sizeof(double)*ARR_SZ, cudaMemcpyHostToDevice);
    //cudaMemcpy(gpu_new, newdata,sizeof(float)*SIZE, cudaMemcpyHostToDevice);
    
    
    //for(i = 1;i <= nsteps; ++i){
    int _num = ARR_SZ/THREAD_NUM;
    if(_num > BLOCK_NUM) _num = BLOCK_NUM;
    kernel_cuda_update<<< _num, THREAD_NUM>>>( gpu_old, gpu_val, gpu_new, 
                                        ARR_SZ, STEP,GRID_SZ,
                                        C, K, dt);
   // }
    cudaMemcpy(olddata, gpu_old,sizeof(double)*ARR_SZ, cudaMemcpyDeviceToHost);
    cudaMemcpy(data, gpu_val,sizeof(double)*ARR_SZ, cudaMemcpyDeviceToHost);
    cudaMemcpy(newdata, gpu_new,sizeof(double)*ARR_SZ, cudaMemcpyDeviceToHost);
    cudaFree(gpu_new);
    cudaFree(gpu_old);
    cudaFree(gpu_val);



}
