CC = gcc
CFLAGS = -O2 -Wall 
LIBS = -lm -lboost_system -lboost_thread
# SEQUENTIAL  - running in sequential (default) 
# THREAD_POOL - running in parallel with a thread pool
# Usage : $make IMP=<MACRO>
IMP ?= SEQUENTIAL

wave: wave_2d.c wave_2d.h wave_2d_thp.o libthp.so 
	$(CC) $^ -o $@ $(CFLAGS) $(LIBS) -DIMP_$(IMP)

wave_2d_thp.o: wave_2d_thp.c 
	$(CC) $(CFLAGS) -c $< -o $@

libthp.so: wave_2d_thp.cpp
	g++ -std=gnu++11 -O2 -fpic -shared -Wall -Wextra $< -Wl,-soname,$@ -o $@