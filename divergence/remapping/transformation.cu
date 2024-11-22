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
        int src = input[index];
        int cond = (src % 10 != 0);
        int dest = 0;
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

void transform(int* input, int* transformed_input, unsigned long long int numberOfElements) {
    unsigned long long int start_index = 0;
    unsigned long long int end_index = numberOfElements - 1;
    for (unsigned long long int i = 0; i < numberOfElements; i++) {
        int value = input[i];
        if (value % 10 != 0) {
            transformed_input[start_index] = value;
            start_index += 1;
        } else {
            transformed_input[end_index] = value;
            end_index -= 1;
        }
    }
}

int main () {
    GPUTimer timer_kernel;
    GPUTimer timer_total;
    
    CPUTimer timer_transform;

    float kernel_sum = 0;
    float total_sum = 0;
    
    double transform_sum = 0;

    for(int j = 0; j < NUMBER_OF_TESTS; j++) {        

        timer_total.start();

        int *input;
        int *h_input, *h_output;
        unsigned long long int numberOfElements = ((DATA_SIZE / 2) * (1L << 30)) / sizeof(int);

        input = (int *) malloc(numberOfElements * sizeof(int));
        h_input = (int *) malloc(numberOfElements * sizeof(int));
        h_output = (int *) malloc(numberOfElements * sizeof(int));
    

        for (unsigned long long int i = 0; i < numberOfElements; i++) {
            input[i] = (int) (i % (1 << 29));
        }

        timer_transform.start();

        transform(input, h_input, numberOfElements);

        timer_transform.end();
        transform_sum += timer_transform.elapsed();

        int *d_input, *d_output;

        cudaMalloc((void **) &d_input, numberOfElements * sizeof(int));
        cudaMalloc((void **) &d_output, numberOfElements * sizeof(int));

        cudaMemcpy((void *) d_input, (void *) h_input, numberOfElements * sizeof(int), cudaMemcpyHostToDevice);

        timer_kernel.start();

        int blockSize = 256;
        int numBlocks = 196608 / blockSize;
        kernel<<<numBlocks, blockSize>>>(d_input, d_output, numberOfElements);
        cudaDeviceSynchronize();

        timer_kernel.end();
        kernel_sum += timer_kernel.elapsed();


        cudaMemcpy((void *) h_output, (void *) d_output, numberOfElements * sizeof(int), cudaMemcpyDeviceToHost);

        free(h_input);
        free(h_output);

        cudaFree(d_input);
        cudaFree(d_output);

        timer_total.end();
        total_sum += timer_total.elapsed();

    }

    printf("Average transform time for size %ldGB is %f\n", DATA_SIZE, transform_sum / NUMBER_OF_TESTS);
    printf("Average kernel time for size %ldGB is %f\n", DATA_SIZE, kernel_sum / NUMBER_OF_TESTS);
    printf("Average total time for size %ldGB is %f\n", DATA_SIZE, total_sum / NUMBER_OF_TESTS);

    return 0;
}