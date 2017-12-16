CC = nvcc

wave: wave_2d.cu wave_2d.h
	$(CC) $< -o $@ $(CFLAGS)
