#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "gputimer.cuh"
#include "cputimer.cuh"

#define NUMBER_OF_TESTS 10
#define DATA_SIZE 8L
#define ITERATIONS 10000


__global__ void kernel(int* input, int* output, unsigned long long int numberOfElements) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int numberOfTotalThreads = (unsigned long long int) gridDim.x * blockDim.x;
    unsigned long long int numberOfWorks = numberOfElements / numberOfTotalThreads;
    for (unsigned long long int i = 0; i < numberOfWorks; ++i) {
        unsigned long long int index = idx * numberOfWorks + i;
        int cond = (index % 10 != 0);
        int dest = 0;
        int src = input[index];
        int j;
        if (cond) {
            for (j = 0; j < ITERATIONS; j++) {
                dest += src * src + 5 * src + 6;
                dest *= 3;
                dest += src * 2;
            }
        }
        else {
            for (j = 0; j < ITERATIONS; j++) {
                dest += src * src + src * 2 + 1;
                dest *= 3;
                dest += src * 3;
            }
        }
        output[index] = dest;
    }
}

int main () {
    
    GPUTimer timer_cudaMalloc;
    GPUTimer timer_HD;
    GPUTimer timer_kernel;
    GPUTimer timer_DH;
    GPUTimer timer_cudaFree;
    GPUTimer timer_total;

    CPUTimer timer_malloc;
    CPUTimer timer_init;
    CPUTimer timer_free;


    float cudaMalloc_sum = 0;
    float HD_sum = 0;
    float kernel_sum = 0;
    float DH_sum = 0;
    float cudaFree_sum = 0;
    float total_sum = 0;

    double malloc_sum = 0;
    double init_sum = 0;
    double free_sum = 0;

    for(int j = 0; j < NUMBER_OF_TESTS; j++) {        

        timer_total.start();

        int *h_input, *h_output;
        unsigned long long int numberOfElements = ((DATA_SIZE / 2) * (1L << 30)) / sizeof(int);

        timer_malloc.start();


        h_input = (int *) malloc(numberOfElements * sizeof(int));
        h_output = (int *) malloc(numberOfElements * sizeof(int));

        timer_malloc.end();
        malloc_sum += timer_malloc.elapsed();
    
        timer_init.start();

        for (unsigned long long int i = 0; i < numberOfElements; i++) {
            h_input[i] = (int) (i % (1 << 29));
        }

        timer_init.end();
        init_sum += timer_init.elapsed();

        int *d_input, *d_output;

        timer_cudaMalloc.start();

        cudaMalloc((void **) &d_input, numberOfElements * sizeof(int));
        cudaMalloc((void **) &d_output, numberOfElements * sizeof(int));

        timer_cudaMalloc.end();
        cudaMalloc_sum += timer_cudaMalloc.elapsed();

        timer_HD.start();

        cudaMemcpy((void *) d_input, (void *) h_input, numberOfElements * sizeof(int), cudaMemcpyHostToDevice);

        timer_HD.end();
        HD_sum += timer_HD.elapsed();

        timer_kernel.start();

        int blockSize = 256;
        int numBlocks = 196608 / blockSize;
        kernel<<<numBlocks, blockSize>>>(d_input, d_output, numberOfElements);
        cudaDeviceSynchronize();

        timer_kernel.end();
        kernel_sum += timer_kernel.elapsed();

        timer_DH.start();

        cudaMemcpy((void *) h_output, (void *) d_output, numberOfElements * sizeof(int), cudaMemcpyDeviceToHost);

        timer_DH.end();
        DH_sum += timer_DH.elapsed();

        timer_free.start();

        free(h_input);
        free(h_output);

        timer_free.end();
        free_sum += timer_free.elapsed();

        timer_cudaFree.start();

        cudaFree(d_input);
        cudaFree(d_output);

        timer_cudaFree.end();
        cudaFree_sum += timer_cudaFree.elapsed();

        timer_total.end();
        total_sum += timer_total.elapsed();

    }

    printf("Average malloc time for size %ldGB is %f\n", DATA_SIZE, malloc_sum / NUMBER_OF_TESTS);
    printf("Average init time for size %ldGB is %f\n", DATA_SIZE, init_sum / NUMBER_OF_TESTS);
    printf("Average cudaMalloc time for size %ldGB is %f\n", DATA_SIZE, cudaMalloc_sum / NUMBER_OF_TESTS);
    printf("Average HD time for size %ldGB is %f\n", DATA_SIZE, HD_sum / NUMBER_OF_TESTS);
    printf("Average kernel time for size %ldGB is %f\n", DATA_SIZE, kernel_sum / NUMBER_OF_TESTS);
    printf("Average DH time for size %ldGB is %f\n", DATA_SIZE, DH_sum / NUMBER_OF_TESTS);
    printf("Average free time for size %ldGB is %f\n", DATA_SIZE, free_sum / NUMBER_OF_TESTS);
    printf("Average cudaFree time for size %ldGB is %f\n", DATA_SIZE, cudaFree_sum / NUMBER_OF_TESTS);
    printf("Average total time for size %ldGB is %f\n", DATA_SIZE, total_sum / NUMBER_OF_TESTS);

    return 0;
}