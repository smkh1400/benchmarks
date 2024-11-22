#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "gputimer.cuh"
#include "cputimer.cuh"

#define NUMBER_OF_TESTS 1
#define DATA_SIZE 20L

__global__ void kernel(int* input, unsigned long long int numberOfElements) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int stride = (unsigned long long int) gridDim.x * blockDim.x;
    for (unsigned long long int i = idx; i < numberOfElements; i += stride) {
        input[i] *= 2;
    }
}

int main() {
    
    //Define Timers


    GPUTimer timer_kernel;

    float kernel_sum = 0;
    
    unsigned long long int number_of_elements = (DATA_SIZE * (1L << 30)) / sizeof(int);

    for (int j = 0; j < NUMBER_OF_TESTS; j++) {

        // Allocate host array

        int* h_input = (int*) malloc (number_of_elements * sizeof(int));

        // Initialize array on the host

        for (unsigned long long int i = 0; i < number_of_elements; i++) {
            h_input[i] = (int)(i % (1 << 29));
        }

        // Allocate device memory
        int* d_input;

        cudaMalloc((void**)&d_input, number_of_elements * sizeof(int));

        // Copy host arrays to device

        cudaMemcpy(d_input, h_input, number_of_elements * sizeof(int), cudaMemcpyHostToDevice);
        

        // Define block and grid sizes
        int blockSize = 64;
        int numBlocks = 196608 / blockSize;

        
        // Measure time for unaligned array processing
        timer_kernel.start();

        
        kernel<<<numBlocks, blockSize>>>(d_input, number_of_elements);

        timer_kernel.end();
        kernel_sum += timer_kernel.elapsed();

        // Copy device arrays to host

        cudaMemcpy(h_input, d_input, number_of_elements * sizeof(int), cudaMemcpyDeviceToHost);

        // Clean up

        cudaFree(d_input);

        free(h_input);
    }

    printf("For %ldGB elements and %d tests {\n", DATA_SIZE, NUMBER_OF_TESTS);
    printf("Average kernel time : coalesced -> %lf\n", kernel_sum / NUMBER_OF_TESTS);
    printf("}\n");

}