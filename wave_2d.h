#ifndef _WAVE_2D_H_
#define _WAVE_2D_H_

#define STEPS 1000

#ifdef _DATA_100_
#define GRID_SZ 100
#define ARR_SZ GRID_SZ * GRID_SZ
#define PEAK_SZ 90
#define PEAK_POS 5
#endif

#ifdef _DATA_500_
#define GRID_SZ 500
#define ARR_SZ GRID_SZ * GRID_SZ
#define PEAK_SZ 400
#define PEAK_POS 50
#endif

#ifdef _DATA_1000_
#define GRID_SZ 1000
#define ARR_SZ GRID_SZ * GRID_SZ
#define PEAK_SZ 900
#define PEAK_POS 50
#endif

#ifdef _DATA_5000_
#define GRID_SZ 5000
#define ARR_SZ GRID_SZ * GRID_SZ
#define PEAK_SZ 4000
#define PEAK_POS 500
#endif

#ifdef _DATA_10000_
#define GRID_SZ 10000
#define ARR_SZ GRID_SZ * GRID_SZ
#define PEAK_SZ 9000
#define PEAK_POS 500
#endif


void sequential_update(double *data, double *olddata, double *newdata, double, double, double );
void threadpool_update(double *data, double *olddata, double *newdata, double C, double K, double dt);

#endif
