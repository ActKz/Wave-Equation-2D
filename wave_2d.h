#ifndef _WAVE_2D_H_
#define _WAVE_2D_H_

#define GRID_SZ 10000
#define ARR_SZ GRID_SZ * GRID_SZ
#define PEAK_SZ 9001

void sequential_update(double *data, double *olddata, double *newdata, double, double, double );
void threadpool_update(double *data, double *olddata, double *newdata, double C, double K, double dt);

#endif
