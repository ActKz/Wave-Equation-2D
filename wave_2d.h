#define GRID_SZ 100
#define ARR_SZ GRID_SZ * GRID_SZ
#define PEAK_SZ 31
#define STEP 20
#define THREAD_NUM 512
#define BLOCK_NUM 256


void sequential_update(double *data, double *olddata, double *newdata, double, double, double );
void cuda_update(double *data, double *olddata, double *newdata, double, double, double );
