CC = gcc
CFLAGS = -O2 -Wall -lm


wave: wave_2d.c wave_2d.h
	$(CC) $< -o $@ $(CFLAGS)
