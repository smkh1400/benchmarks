#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "gputimer.cuh"
#include "cputimer.cuh"

#define NUMBER_OF_TESTS 10
#define DATA_SIZE 20L

// __global__ void kernel(int* input, unsigned long long int numberOfElements) {
//     unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned long long int numberOfTotalThreads = (unsigned long long int) gridDim.x * blockDim.x;
//     unsigned long long int numberOfWorks = numberOfElements / numberOfTotalThreads;
//     for (unsigned long long int i = 0; i < numberOfWorks; i += 1) {
//         unsigned long long int index = idx * numberOfWorks + i;
//         input[index] = input[index] * 2;
//     }
// }

__global__ void kernel(int* input, unsigned long long int numberOfElements) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int stride = (unsigned long long int) gridDim.x * blockDim.x;
    for (unsigned long long int i = idx; i < numberOfElements; i += stride) {
        input[i] = input[i] * 2;
    }
}

int main () {

    GPUTimer timer_total;
    GPUTimer timer_HD;
    GPUTimer timer_DH;
    GPUTimer timer_kernel;
    
    CPUTimer timer_init;
    CPUTimer timer_malloc;
    CPUTimer timer_free;
    

    float total_sum = 0;
    float HD_sum = 0;
    float kernel_sum = 0;
    float DH_sum = 0;

    double init_sum = 0;
    double malloc_sum = 0;
    double free_sum = 0;

    for(int j = 0; j < NUMBER_OF_TESTS; j++) {

        timer_total.start();

        int *h_input;
        unsigned long long int numberOfElements = (DATA_SIZE * (1L << 30)) / sizeof(int);

        timer_malloc.start();

        h_input = (int *) malloc(numberOfElements * sizeof(int));

        timer_malloc.end();
        malloc_sum += timer_malloc.elapsed();

        timer_init.start();

        for (unsigned long long int i = 0; i < numberOfElements; i++) {
            h_input[i] = (int) (i % (1 << 29));
        }

        timer_init.end();
        init_sum += timer_init.elapsed();

        int *d_input;

        cudaMalloc((void **) &d_input, numberOfElements * sizeof(int));

        timer_HD.start();

        cudaMemcpy((void *) d_input, (void *) h_input, numberOfElements * sizeof(int), cudaMemcpyHostToDevice);

        timer_HD.end();
        HD_sum += timer_HD.elapsed();

        timer_kernel.start();

        int blockSize = 256;
        int numBlocks = 196608 / blockSize;
        kernel<<<numBlocks, blockSize>>>(d_input, numberOfElements);
        cudaDeviceSynchronize();

        timer_kernel.end();
        kernel_sum += timer_kernel.elapsed();


        timer_DH.start();

        cudaMemcpy((void *) h_input, (void *) d_input, numberOfElements * sizeof(int), cudaMemcpyDeviceToHost);


        timer_DH.end();
        DH_sum += timer_DH.elapsed();

        timer_free.start();



        free(h_input);

        timer_free.end();
        free_sum += timer_free.elapsed();

        cudaFree(d_input);

        timer_total.end();
        total_sum += timer_total.elapsed();


    }

    printf("Average malloc time for size %ldGB is %lf\n", DATA_SIZE, malloc_sum / NUMBER_OF_TESTS);
    printf("Average init time for size %ldGB is %lf\n", DATA_SIZE, init_sum / NUMBER_OF_TESTS);
    printf("Average HD time for size %ldGB is %f\n", DATA_SIZE, HD_sum / NUMBER_OF_TESTS);
    printf("Average kernel time for size %ldGB is %f\n", DATA_SIZE, kernel_sum / NUMBER_OF_TESTS);
    printf("Average DH time for size %ldGB is %f\n", DATA_SIZE, DH_sum / NUMBER_OF_TESTS);
    printf("Average free time for size %ldGB is %lf\n", DATA_SIZE, free_sum / NUMBER_OF_TESTS);
    printf("Average total time for size %ldGB is %f\n", DATA_SIZE, total_sum / NUMBER_OF_TESTS);

    return 0;
}