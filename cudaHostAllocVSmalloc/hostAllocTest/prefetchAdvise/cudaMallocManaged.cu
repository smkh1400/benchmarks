#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "gputimer.cuh"
#include "cputimer.cuh"
#include <math.h>

#define NUMBER_OF_TESTS 1
#define DATA_SIZE 20L
#define BATCH_DATA_SIZE 2L
#define BLOCK_SIZE 256
#define MAX_THREAD 196608

__global__ void kernel(int* input, unsigned long long int numberOfElements) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int stride = (unsigned long long int) gridDim.x * blockDim.x;
    for (unsigned long long int i = idx; i < numberOfElements; i += stride) {
        input[i] *= 2;
    }
}

__global__ void mediumKernel(int* input, unsigned long long int numberOfElements) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int numberOfTotalThreads = (unsigned long long int) gridDim.x * blockDim.x;
    unsigned long long int numberOfWorks = numberOfElements / numberOfTotalThreads;

    for (unsigned long long int i = 0; i < numberOfWorks; i += 1) {
        unsigned long long int index = idx * numberOfWorks + i;

        // Original computation
        int value = input[index] * 2;

        // Additional computations
        value = value + (value / 3) - (value % 5);
        value = (value * value) / 2;

        // Incorporate trigonometric function
        float sinValue = sinf(value);
        float cosValue = cosf(value);
        float tanValue = tanf(value);
        float combinedValue = sinValue + cosValue + tanValue;

        // More complex arithmetic operations
        for (int j = 0; j < 5; j++) {
            combinedValue += sqrtf(value + j) * logf(value + j + 1);
        }

        // Final assignment using C-style cast
        input[index] = (int)combinedValue;
    }
}


__global__ void complexKernel(int* input, unsigned long long int numberOfElements) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int numberOfTotalThreads = (unsigned long long int) gridDim.x * blockDim.x;
    unsigned long long int numberOfWorks = numberOfElements / numberOfTotalThreads;

    for (unsigned long long int i = 0; i < numberOfWorks; i += 1) {
        unsigned long long int index = idx * numberOfWorks + i;

        // Original computation
        int value = input[index] * 2;

        // Additional complex computations
        // Polynomial evaluation (e.g., x^3 + 2x^2 - 3x + 5)
        int polynomialResult = value * value * value + 2 * value * value - 3 * value + 5;

        // Complex arithmetic involving trigonometric and logarithmic functions
        float trigValue = sinf(polynomialResult) + cosf(polynomialResult) - tanf(polynomialResult);
        // float logValue = logf(abs(polynomialResult) + 1.0f);  // Avoid log(0) by adding 1

        // Nested loop for further complexity
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 5; k++) {
                // Simulate a matrix-like operation
                trigValue += sinf(polynomialResult + j * k) * cosf(polynomialResult - j * k);
                // logValue += sqrtf(abs(polynomialResult + j - k) + 1.0f) * logf(abs(polynomialResult + k) + 1.0f);
            }
        }

        // Combined result with more arithmetic operations
        // float combinedValue = trigValue + logValue * 2.0f + powf(polynomialResult, 2.0f);
        // float combinedValue = logValue * 2.0f + powf(polynomialResult, 2.0f);
        float combinedValue = trigValue  + powf(polynomialResult, 2.0f);

        // Final assignment using C-style cast
        input[index] = (int)combinedValue;
    }
}



int main () {

    GPUTimer timer_total;
    GPUTimer timer_malloc;
    GPUTimer timer_kernel;
    GPUTimer timer_free;

    
    CPUTimer timer_init;
    

    float total_sum = 0;
    float kernel_sum = 0;
    float malloc_sum = 0;
    float free_sum = 0;

    double init_sum = 0;

    for(int j = 0; j < NUMBER_OF_TESTS; j++) {

        timer_total.start();

        int *h_input, *h_other_input;

        unsigned long long int numberOfElements = ((DATA_SIZE / 2) * (1L << 30)) / sizeof(int);
        unsigned long long int numberOfBatchElements = ((BATCH_DATA_SIZE / 2) * (1L << 30) / sizeof(int));

        timer_malloc.start();

        cudaMallocManaged((void **) &h_input, numberOfElements * sizeof(int));
        cudaMallocManaged((void **) &h_other_input, numberOfElements * sizeof(int));

        timer_malloc.end();
        malloc_sum += timer_malloc.elapsed();


        timer_init.start();

        for (unsigned long long int i = 0; i < numberOfElements; i++) {
            h_input[i] =  (int) (i % (1 << 29));
            h_other_input[i] = (int) ((i + numberOfElements) % (1 << 29));
        }

        timer_init.end();
        init_sum += timer_init.elapsed();

        cudaStream_t stream1;
        cudaStream_t stream2;

        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);



        for (int i = 0; i < DATA_SIZE; i += BATCH_DATA_SIZE) {
            
            unsigned long long int offset = (i / 2) * (1L << 30) / sizeof(int);

            timer_kernel.start();

            int blockSize = BLOCK_SIZE;
            int numBlocks = MAX_THREAD / blockSize;
            kernel<<<numBlocks / 2, blockSize, 0, stream1>>>(h_input + offset, numberOfBatchElements);
            kernel<<<numBlocks / 2, blockSize, 0, stream2>>>(h_other_input + offset, numberOfBatchElements);

            timer_kernel.end();
            kernel_sum += timer_kernel.elapsed();
        }



        timer_free.start();



        cudaFree(h_input);
        cudaFree(h_other_input);
        
        timer_free.end();
        free_sum += timer_free.elapsed();


        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);

        timer_total.end();
        total_sum += timer_total.elapsed();

    }

    printf("Average mallocManaged time for size %ldGB is %f\n", DATA_SIZE, malloc_sum / NUMBER_OF_TESTS);
    printf("Average init time for size %ldGB is %lf\n", DATA_SIZE, init_sum / NUMBER_OF_TESTS);
    printf("Average kernel time for size %ldGB is %f\n", DATA_SIZE, kernel_sum / NUMBER_OF_TESTS);
    printf("Average free time for size %ldGB is %f\n", DATA_SIZE, free_sum / NUMBER_OF_TESTS);
    printf("Average total time for size %ldGB is %f\n", DATA_SIZE, total_sum / NUMBER_OF_TESTS);

    return 0;
}