#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "gputimer.cuh"
#include "cputimer.cuh"

#define NUMBER_OF_TESTS 10
#define DATA_SIZE 20L
#define ALIGNMENT_SIZE  8
#define VARIABLE_COUNT 2

typedef struct __align__(ALIGNMENT_SIZE) {
    unsigned int list[VARIABLE_COUNT];
} AlignedData;

__global__ void alignedKernel (AlignedData* array, unsigned long long int numberOfElements) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int numberOfTotalThreads = gridDim.x * blockDim.x;
    unsigned long long int numberOfWorks = numberOfElements / numberOfTotalThreads;
    for (unsigned long long int i = 0; i < numberOfWorks; i += 1) {
        unsigned long long int index = idx * numberOfWorks + i;
        for (int j = 0; j < VARIABLE_COUNT; j++) {
            array[index].list[j] *= 2;
        }
    }
}

int main() {
    
    //Define Timers
    CPUTimer timer_aligned_malloc;
    CPUTimer timer_aligned_init;
    CPUTimer timer_aligned_free;


    GPUTimer timer_aligned_cudaMalloc;
    GPUTimer timer_aligned_HD;
    GPUTimer timer_aligned_kernel;
    GPUTimer timer_aligned_DH;
    GPUTimer timer_aligned_cudaFree;

    double aligned_malloc_sum = 0;
    double aligned_init_sum = 0;
    double aligned_free_sum = 0;

    float aligned_kernel_sum = 0;
    float aligned_cudaMalloc_sum = 0;
    float aligned_HD_sum = 0;
    float aligned_DH_sum = 0;
    float aligned_cudaFree_sum = 0;

    unsigned long long int number_of_elements = (DATA_SIZE * (1L << 30)) / sizeof(AlignedData);

    for (int j = 0; j < NUMBER_OF_TESTS; j++) {

        // Allocate host arrays

        timer_aligned_malloc.start();

        AlignedData* alignedHostArray = (AlignedData*) malloc (number_of_elements * sizeof(AlignedData));

        timer_aligned_malloc.end();
        aligned_malloc_sum += timer_aligned_malloc.elapsed();

        // Initialize arrays on the host

        timer_aligned_init.start();

        for (unsigned long long int i = 0; i < number_of_elements; i++) {
            for (int k = 0; k < VARIABLE_COUNT; k++) {
                alignedHostArray[i].list[k] = k;
            }
        }

        timer_aligned_init.end();
        aligned_init_sum += timer_aligned_init.elapsed();

        // Allocate device memory
        AlignedData* d_alignedArray;

        timer_aligned_cudaMalloc.start();

        cudaMalloc((void**)&d_alignedArray, number_of_elements * sizeof(AlignedData));

        timer_aligned_cudaMalloc.end();
        aligned_cudaMalloc_sum += timer_aligned_cudaMalloc.elapsed();

        // Copy host arrays to device

        timer_aligned_HD.start();

        cudaMemcpy(d_alignedArray, alignedHostArray, number_of_elements * sizeof(AlignedData), cudaMemcpyHostToDevice);

        timer_aligned_HD.end();
        aligned_HD_sum += timer_aligned_HD.elapsed();

        // Define block and grid sizes
        int blockSize = 256;
        int numBlocks = 196608 / blockSize;

        
        // Measure time for unaligned array processing

        timer_aligned_kernel.start();

        alignedKernel<<<numBlocks, blockSize>>>(d_alignedArray, number_of_elements);
        
        timer_aligned_kernel.end();
        aligned_kernel_sum += timer_aligned_kernel.elapsed();

        // Copy device arrays to host

        timer_aligned_DH.start();

        cudaMemcpy(alignedHostArray, d_alignedArray, number_of_elements * sizeof(AlignedData), cudaMemcpyDeviceToHost);

        timer_aligned_DH.end();
        aligned_DH_sum += timer_aligned_DH.elapsed();


        // Clean up

        timer_aligned_cudaFree.start();

        cudaFree(d_alignedArray);

        timer_aligned_cudaFree.end();
        aligned_cudaFree_sum += timer_aligned_cudaFree.elapsed();

        timer_aligned_free.start();

        free(alignedHostArray);

        timer_aligned_free.end();
        aligned_free_sum += timer_aligned_free.elapsed();

    }

    printf("For %ldGB elements each %dbytes with %d alignment and %d tests {\n", DATA_SIZE, VARIABLE_COUNT * 4, ALIGNMENT_SIZE, NUMBER_OF_TESTS);
    printf("Average malloc time aligned -> %lf\n", aligned_malloc_sum / NUMBER_OF_TESTS);
    printf("Average cudaMalloc time aligned -> %lf\n", aligned_cudaMalloc_sum / NUMBER_OF_TESTS);
    printf("Average init time aligned -> %lf\n", aligned_init_sum / NUMBER_OF_TESTS);
    printf("Average HD time aligned -> %lf\n", aligned_HD_sum / NUMBER_OF_TESTS);
    printf("Average kernel time aligned -> %lf\n", aligned_kernel_sum / NUMBER_OF_TESTS);
    printf("Average DH time aligned -> %lf\n", aligned_DH_sum / NUMBER_OF_TESTS);
    printf("Average cudaFree time aligned -> %lf\n", aligned_cudaFree_sum / NUMBER_OF_TESTS);
    printf("Average free time aligned -> %lf\n", aligned_free_sum / NUMBER_OF_TESTS);
    printf("}\n");

}