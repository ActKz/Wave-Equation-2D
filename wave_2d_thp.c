#include "wave_2d.h"

extern void _threadpool_update_cpp(double* data, double* olddata, double* newdata, double C, double K, double dt);

void threadpool_update(double* data, double* olddata, double* newdata, double C, double K, double dt)
{
    _threadpool_update_cpp(data, olddata, newdata, C, K, dt);
}