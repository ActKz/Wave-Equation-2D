#ifndef WAVE_2D_H
#define WAVE_2D_H

#define GRID_SZ 100
#define ARR_SZ GRID_SZ * GRID_SZ
#define PEAK_SZ 31

void sequential_update(double *data, double *olddata, double *newdata, double, double, double );

void threadpool_update(double *data, double *olddata, double *newdata, double C, double K, double dt);

#endif 