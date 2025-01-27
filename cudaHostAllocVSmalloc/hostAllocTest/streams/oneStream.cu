#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "gputimer.cuh"
#include "cputimer.cuh"
#include <math.h>

#define NUMBER_OF_TESTS 10
#define DATA_SIZE 12L
#define BATCH_DATA_SIZE 3L
#define BLOCK_SIZE 256
#define MAX_THREAD 196608

bool areAllTrue(bool* array, int size) {
    for (int i = 0; i < size; i++) {
        if (!array[i]) { // If any element is false
            return false;
        }
    }
    return true; // All elements are true
}

__global__ void kernel(int* input, unsigned long long int numberOfElements, bool* checks) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    checks[idx] = false;
    __syncthreads();
    unsigned long long int numberOfTotalThreads = (unsigned long long int) gridDim.x * blockDim.x;
    unsigned long long int numberOfWorks = numberOfElements / numberOfTotalThreads;
    for (unsigned long long int i = 0; i < numberOfWorks; i += 1) {
        unsigned long long int index = idx * numberOfWorks + i;
        input[index] = input[index] * 2;
    }
    checks[idx] = true;
}

__global__ void mediumKernel(int* input, unsigned long long int numberOfElements, bool* checks) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    checks[idx] = false;
    __syncthreads();
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
    checks[idx] = true;
}

__global__ void complexKernel(int* input, unsigned long long int numberOfElements, bool* checks) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    checks[idx] = false;
    __syncthreads();
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
        float logValue = logf(abs(polynomialResult) + 1.0f);  // Avoid log(0) by adding 1

        // Nested loop for further complexity
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 5; k++) {
                // Simulate a matrix-like operation
                trigValue += sinf(polynomialResult + j * k) * cosf(polynomialResult - j * k);
                logValue += sqrtf(abs(polynomialResult + j - k) + 1.0f) * logf(abs(polynomialResult + k) + 1.0f);
            }
        }

        // Combined result with more arithmetic operations
        float combinedValue = trigValue + logValue * 2.0f + powf(polynomialResult, 2.0f);

        // Final assignment using C-style cast
        input[index] = (int)combinedValue;
    }
    checks[idx] = true;
}



int main () {

    GPUTimer timer_total;
    GPUTimer timer_HD;
    GPUTimer timer_DH;
    GPUTimer timer_kernel;

    CPUTimer timer_free;
    

    float total_sum = 0;
    float HD_sum = 0;
    float kernel_sum = 0;
    float DH_sum = 0;

    double free_sum = 0;

    for(int j = 0; j < NUMBER_OF_TESTS; j++) {


        int *h_input;
        bool *h_checks;

        unsigned long long int numberOfElements = ((DATA_SIZE) * (1L << 30)) / sizeof(int);
        unsigned long long int numberOfBatchElements = ((BATCH_DATA_SIZE) * (1L << 30) / sizeof(int));

        cudaHostAlloc((void **) &h_input, numberOfElements * sizeof(int), cudaHostAllocDefault);

        cudaHostAlloc((void **) &h_checks, MAX_THREAD * sizeof(bool), cudaHostAllocDefault);
        

        for (unsigned long long int i = 0; i < numberOfElements; i++) {
            h_input[i] = (int) (i % (1 << 29));
        }
        


        int *d_input;
        bool *d_checks;

        cudaMalloc((void **) &d_input, numberOfBatchElements * sizeof(int));
        cudaMalloc((void **) &d_checks, MAX_THREAD * sizeof(bool));

        timer_total.start();
        // timer_HD.start();

        for (int i = 0; i < DATA_SIZE; i += BATCH_DATA_SIZE) {

            unsigned long long int offset = i * (1L << 30) / sizeof(int);

            cudaMemcpy((void *) d_input, (void *) (h_input + offset), numberOfBatchElements * sizeof(int), cudaMemcpyHostToDevice);

            // timer_HD.end();
            // HD_sum += timer_HD.elapsed();


            int blockSize = BLOCK_SIZE;
            int numBlocks = MAX_THREAD / blockSize;
            
            // timer_kernel.start();

            mediumKernel<<<numBlocks, blockSize>>>(d_input, numberOfBatchElements, d_checks);
            
            // timer_kernel.end();
            // kernel_sum += timer_kernel.elapsed();
            


            // timer_DH.start();

            cudaMemcpy((void *) (h_input + offset), (void *) d_input, numberOfBatchElements * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy((void *) (h_checks), (void *) d_checks, MAX_THREAD * sizeof(bool), cudaMemcpyDeviceToHost);

            // cudaDeviceSynchronize();

            // if (!areAllTrue(h_checks, MAX_THREAD)) {
            //     printf("kernel failed in batch %d\n", i);
            // }

            // timer_DH.end();
            // DH_sum += timer_DH.elapsed();
        }

        timer_total.end();
        total_sum += timer_total.elapsed();


        // timer_free.start();

        cudaFreeHost(h_input);
        cudaFreeHost(h_checks);


        cudaFree(d_input);
        cudaFree(h_input);

        // timer_free.end();
        // free_sum += timer_free.elapsed();



    }

    // printf("Average HD time for size %ldGB is %f\n", DATA_SIZE, HD_sum / NUMBER_OF_TESTS);
    // printf("Average kernel time for size %ldGB is %f\n", DATA_SIZE, kernel_sum / NUMBER_OF_TESTS);
    // printf("Average DH time for size %ldGB is %f\n", DATA_SIZE, DH_sum / NUMBER_OF_TESTS);
    // printf("Average free time for size %ldGB is %lf\n", DATA_SIZE, free_sum / NUMBER_OF_TESTS);
    printf("Average total time for size %ldGB is %f\n", DATA_SIZE, total_sum / NUMBER_OF_TESTS);

    return 0;
}