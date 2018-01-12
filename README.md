# Wave Equation 2D

This project implement the model of 2-dimension water wave simulator with three version:

- Sequential
- Thread pool (using boost library)
- CUDA

# Data size

- 100^2
- 500^2
- 1000^2
- 5000^2
- 10000^2

# Usage

Don't print out data

```
make SIZE=100
./bin/100_wave_seq
./bin/100_wave_cuda
./bin/100_wave_threadpool
```

Print out data

```
make SIZE=100 PRINT=PRINT
./bin/100_wave_seq
./bin/100_wave_cuda
./bin/100_wave_threadpool
```

# TODO: Add simulation to display wave animation
