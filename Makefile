CC = gcc
NVCC = nvcc
CFLAGS = -O2 -lm
LIBS = -lboost_system -lboost_thread

wave_threadpool: wave_2d.c wave_2d.h wave_2d_thp.o libthp.so
	$(CC) $^ -o $@ $(CFLAGS) $(LIBS) -D _WAVE_THREADPOOL_

wave_seq: wave_2d.c wave_2d.h main.c
	$(CC) $^ -o bin/$@ $(CFLAGS) -D _WAVE_SEQUENTIAL_

wave_cuda: wave_2d.cu main.c
	$(NVCC) $^ -o bin/$@ -Xcompiler $(CFLAGS) -D _WAVE_CUDA_ -D_FORCE_INLINES

wave_2d_thp.o: wave_2d_thp.c
	$(CC) $(CFLAGS) -c $< -o $@

libthp.so: wave_2d_thp.cpp
	g++ -std=gnu++11 -O2 -fpic -shared -Wall -Wextra $< -Wl,-soname,$@ -o $@

clean:
	rm *.o bin/*
