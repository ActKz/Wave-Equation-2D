CC = gcc
CFLAGS = -O2 -Wall -lm
IMP ?= SEQUENTIAL


wave_seq: wave_2d.c wave_2d.h main.c
	$(CC) $^ -o $@ $(CFLAGS) -D $(IMP)

clean:
	rm wave_seq
