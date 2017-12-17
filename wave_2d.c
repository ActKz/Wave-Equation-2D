#include "wave_2d.h"
#include <math.h>


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
