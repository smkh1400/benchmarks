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
    unsigned long long int numberOfTotalThreads = (unsigned long long int) gridDim.x * blockDim.x;
    unsigned long long int numberOfWorks = numberOfElements / numberOfTotalThreads;

    __syncthreads();
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
    checks[idx] = true;
}

__global__ void complexKernel2(int* input, unsigned long long int numberOfElements, bool* checks) {
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
        // Polynomial evaluation and more complex arithmetic
        int polynomialResult = value * value * value + 2 * value * value - 3 * value + 5;

        // Complex arithmetic involving trigonometric and logarithmic functions
        float trigValue = sinf(polynomialResult) + cosf(polynomialResult) - tanf(polynomialResult);
        float logValue = logf(abs(polynomialResult) + 1.0f); // Avoid log(0) by adding 1

        // Nested loop for further complexity
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 5; k++) {
                // Simulate a matrix-like operation with more complex functions
                trigValue += sinf(polynomialResult + j * k) * cosf(polynomialResult - j * k);
                logValue += sqrtf(abs(polynomialResult + j - k) + 1.0f) * logf(abs(polynomialResult + k) + 1.0f);
            }
        }

        // Advanced non-linear transformations
        float nonLinear = tanhf(trigValue) + expf(logValue / (value + 1.0f));
        
        // Iterative refinement (e.g., like in Jacobi or Gauss-Seidel iterations)
        for (int m = 0; m < 15; m++) {
            nonLinear -= 0.5f * (sinf(nonLinear + m) * cosf(nonLinear - m)) / (1.0f + expf(-nonLinear));
        }

        // Further complex arithmetic operations
        float combinedValue = nonLinear + logValue * trigValue + powf(polynomialResult, 2.0f);

        // Final assignment with C-style cast
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


        int *h_input, *h_other_input;
        bool *h_checks, *h_other_checks;

        unsigned long long int numberOfElements = ((DATA_SIZE / 2) * (1L << 30)) / sizeof(int);
        unsigned long long int numberOfBatchElements = ((BATCH_DATA_SIZE / 2) * (1L << 30) / sizeof(int));

        cudaHostAlloc((void **) &h_input, numberOfElements * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc((void **) &h_other_input, numberOfElements * sizeof(int), cudaHostAllocDefault);

        cudaHostAlloc((void **) &h_checks, (MAX_THREAD / 2) * sizeof(bool), cudaHostAllocDefault);
        cudaHostAlloc((void **) &h_other_checks, (MAX_THREAD / 2) * sizeof(bool), cudaHostAllocDefault);
        

        for (unsigned long long int i = 0; i < numberOfElements; i++) {
            h_input[i] =  (int) (i % (1 << 29));
            h_other_input[i] = (int) ((i + numberOfElements) % (1 << 29));
        }

        cudaStream_t stream1;
        cudaStream_t stream2;

        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);


        int *d_input, *d_other_input;
        bool *d_checks, *d_other_checks;

        cudaMalloc((void **) &d_input, numberOfBatchElements * sizeof(int));
        cudaMalloc((void **) &d_other_input, numberOfBatchElements * sizeof(int));

        cudaMalloc((void **) &d_checks, (MAX_THREAD / 2) * sizeof(bool));
        cudaMalloc((void **) &d_other_checks, (MAX_THREAD / 2) * sizeof(bool));

        timer_total.start();
        // timer_HD.start();

        for (int i = 0; i < DATA_SIZE; i += BATCH_DATA_SIZE) {

            unsigned long long int offset = (i / 2) * (1L << 30) / sizeof(int);

            cudaMemcpyAsync((void *) d_input, (void *) (h_input + offset), numberOfBatchElements * sizeof(int), cudaMemcpyHostToDevice, stream1);
            cudaMemcpyAsync((void *) d_other_input, (void *) (h_other_input + offset), numberOfBatchElements * sizeof(int), cudaMemcpyHostToDevice, stream2);

            // timer_HD.end();
            // HD_sum += timer_HD.elapsed();

            // timer_kernel.start();

            int blockSize = BLOCK_SIZE;
            int numBlocks = MAX_THREAD / blockSize;
            mediumKernel<<<numBlocks / 2, blockSize, 0, stream1>>>(d_input, numberOfBatchElements, d_checks);
            mediumKernel<<<numBlocks / 2, blockSize, 0, stream2>>>(d_other_input, numberOfBatchElements, d_other_checks);

            // timer_kernel.end();
            // kernel_sum += timer_kernel.elapsed();


            // timer_DH.start();

            cudaMemcpyAsync((void *) (h_input + offset), (void *) d_input, numberOfBatchElements * sizeof(int), cudaMemcpyDeviceToHost, stream1);
            cudaMemcpyAsync((void *) (h_other_input + offset), (void *) d_other_input, numberOfBatchElements * sizeof(int), cudaMemcpyDeviceToHost, stream2);

            cudaMemcpyAsync((void *) (h_checks), (void *) d_checks, (MAX_THREAD / 2) * sizeof(bool), cudaMemcpyDeviceToHost, stream1);
            cudaMemcpyAsync((void *) (h_other_checks), (void *) d_other_checks, (MAX_THREAD / 2) * sizeof(bool), cudaMemcpyDeviceToHost, stream2);

            // cudaStreamSynchronize(stream1);
            // cudaStreamSynchronize(stream2);

            // if (!areAllTrue(h_checks, MAX_THREAD / 2)) {
            //     printf("stream 1 failed in batch %d\n", i);
            // }
            // if (!areAllTrue(h_other_checks, MAX_THREAD / 2)) {
            //     printf("stream 2 failed in batch %d\n", i);
            // }


            // timer_DH.end();
            // DH_sum += timer_DH.elapsed();
        }


        timer_total.end();
        total_sum += timer_total.elapsed();

        // timer_free.start();



        cudaFreeHost(h_input);
        cudaFreeHost(h_other_input);

        cudaFreeHost(h_checks);
        cudaFreeHost(h_other_checks);


        cudaFreeAsync(d_input, stream1);
        cudaFreeAsync(d_other_input, stream2);

        cudaFreeAsync(d_checks, stream1);
        cudaFreeAsync(d_other_checks, stream2);
        
        // timer_free.end();
        // free_sum += timer_free.elapsed();


        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);


    }

    // printf("Average HD time for size %ldGB is %f\n", DATA_SIZE, HD_sum / NUMBER_OF_TESTS);
    // printf("Average kernel time for size %ldGB is %f\n", DATA_SIZE, kernel_sum / NUMBER_OF_TESTS);
    // printf("Average DH time for size %ldGB is %f\n", DATA_SIZE, DH_sum / NUMBER_OF_TESTS);
    // printf("Average free time for size %ldGB is %lf\n", DATA_SIZE, free_sum / NUMBER_OF_TESTS);
    printf("Average total time for size %ldGB is %f\n", DATA_SIZE, total_sum / NUMBER_OF_TESTS);

    return 0;
}