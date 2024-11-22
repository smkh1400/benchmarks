#ifndef CPUTIMER_CUH
#define CPUTIMER_CUH

#include <time.h>

struct CPUTimer {
    struct timespec start_time;
    struct timespec end_time;

    void start() {
        clock_gettime(1, &start_time);
    }

    void end() {
        clock_gettime(1, &end_time);
    }

    double elapsed() {
        double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
        return elapsed_time;
    }

};

#endif