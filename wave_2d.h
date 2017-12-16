#define GRID_SZ 100
#define ARR_SZ GRID_SZ * GRID_SZ
#define PEAK_SZ 31
#define STEP 2000

void sequential_update(float *data, float *olddata, float *newdata, float, float, float );
void cuda_update(float *data, float *olddata, float *newdata, float, float, float );
