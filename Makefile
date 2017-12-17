CC = gcc
NVCC = nvcc
CFLAGS = -O2 -lm
IMP ?= SEQUENTIAL


wave_seq: wave_2d.c wave_2d.h main.c
	$(CC) $^ -o bin/$@ $(CFLAGS) -D _WAVE_$(IMP)_

wave_cuda: wave_2d.cu main.c
	$(NVCC) $^ -o bin/$@ -Xcompiler $(CFLAGS) -D _WAVE_$(IMP)_ -D_FORCE_INLINES

clean:
	rm *.o bin/*

