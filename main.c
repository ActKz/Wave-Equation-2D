#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "wave_2d.h"


const int steps = 1000;
const double dt = 0.04, C = 16, K = 0.1, h = 6;
double *data, *olddata, *newdata, *tmp;

void array_init(){
    int i, j;
    double **x, *linspace, delta = 2.0/(PEAK_SZ-1.0);
    x = (double**)malloc(sizeof(double*)*PEAK_SZ);
    for(i = 0; i < PEAK_SZ; i++)
        x[i] = (double*)malloc(sizeof(double)*PEAK_SZ);
    linspace = (double*)malloc(sizeof(double)*PEAK_SZ);
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
            data[(i+PEAK_POS)*GRID_SZ+j+PEAK_POS] += h * exp( -5 * (pow(x[i][j], 2 ) + pow(x[j][i], 2 )));
        }
    }
    memcpy(olddata, data, sizeof(double)*ARR_SZ);
    for(i = 0; i < PEAK_SZ; i++)
        free(x[i]);
    free(x);
    free(linspace);
}
void print_result(){
    int i, j;
    for(i = 0; i < GRID_SZ; i++){
        for(j = 0; j < GRID_SZ; j++){
            printf("%lf, ", data[i*GRID_SZ+j]);
        }
        printf("\n");
    }
}
int main(){
    int i;
    array_init();
    clock_t begin = clock();
#if defined(_WAVE_CUDA_)
    cuda_update(olddata, data, newdata, C, K, dt, steps );
#elif defined(_WAVE_THREADPOOL_)
    for(i = 0; i < steps; i++){
        threadpool_update(data, olddata, newdata, C, K, dt);
        tmp = olddata;
        olddata = data;
        data = newdata;
        newdata = tmp;
    }
#else
    for(i = 0; i < steps; i++){
        sequential_update( data, olddata, newdata, C, K, dt);
        tmp = olddata;
        olddata = data;
        data = newdata;
        newdata = tmp;
    }
#endif
    clock_t end = clock();
#if defined(_PRINT_RESULT_)
    print_result();
#endif
    printf("Time spent: %lf\n", (double)(end-begin)/CLOCKS_PER_SEC);
    free(data);
    free(olddata);
    free(newdata);
}
