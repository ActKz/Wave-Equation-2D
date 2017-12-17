#ifndef _WAVE_2D_H_
#define _WAVE_2D_H_

#define GRID_SZ 100
#define ARR_SZ GRID_SZ * GRID_SZ
#define PEAK_SZ 31

void sequential_update(double *data, double *olddata, double *newdata, double, double, double );

#endif
