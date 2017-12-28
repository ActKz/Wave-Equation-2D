CC = gcc
NVCC = nvcc
CFLAGS = -O2 -lm
PRINT ?= NOPRINT
SIZE ?= 100
LIBS = -lboost_system -lboost_thread

wave_seq: wave_2d.c wave_2d.h main.c
	$(CC) $^ -o bin/$(SIZE)_$@ $(CFLAGS) -D _WAVE_SEQUENTIAL_ -D _$(PRINT)_RESULT_ -D _DATA_$(SIZE)_

wave_threadpool: main.c wave_2d.c wave_2d.h wave_2d_thp.o libthp.so
	$(CC) $^ -o bin/$(SIZE)_$@ $(CFLAGS) $(LIBS) -D _WAVE_THREADPOOL_ -D _$(PRINT)_RESULT_ -D _DATA_$(SIZE)_

wave_cuda: wave_2d.cu main.c
	$(NVCC) $^ -o bin/$(SIZE)_$@ -Xcompiler $(CFLAGS) -D _WAVE_CUDA_ -D_FORCE_INLINES -D _$(PRINT)_RESULT_ -D _DATA_$(SIZE)_

wave_2d_thp.o: wave_2d_thp.c
	$(CC) $(CFLAGS) -c $< -o $@ -D _$(PRINT)_RESULT_ -D _DATA_$(SIZE)_

libthp.so: wave_2d_thp.cpp
	g++ -std=gnu++11 -O2 -fpic -shared -Wall -Wextra $< -Wl,-soname,$@ -o $@ -D_DATA_$(SIZE)_ -D _$(PRINT)_RESULT_ -D _DATA_$(SIZE)_

clean:
	rm -f *.so *.o bin/*
