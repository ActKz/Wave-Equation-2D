CC = gcc
CFLAGS = -O2 -Wall -lm


wave_seq: wave_2d.c wave_2d.h main.c
	$(CC) $^ -o $@ $(CFLAGS)

clean:
	rm wave_seq
