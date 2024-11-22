#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "gputimer.cuh"
#include "cputimer.cuh"

#define NUMBER_OF_TESTS 1
#define DATA_SIZE 20L

typedef struct {
    unsigned char a;
    unsigned char b;
    unsigned short int c;
    // unsigned int e;
    // unsigned int f;
    // unsigned int g;
    // unsigned int h;
} UnalignedData;


__global__ void unalignedKernel (UnalignedData* array, unsigned long long int numberOfElements) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int numberOfTotalThreads = gridDim.x * blockDim.x;
    unsigned long long int numberOfWorks = numberOfElements / numberOfTotalThreads;
    for (unsigned long long int i = 0; i < numberOfWorks; i += 1) {
        unsigned long long int index = idx * numberOfWorks + i;
        array[index].a = array[index].a * 2;
        array[index].b = array[index].b * 2;
        array[index].c = array[index].c * 2;
        // array[index].d = array[index].d * 2;
        // array[index].e = array[index].e * 2;
        // array[index].f = array[index].f * 2;
        // array[index].g = array[index].g * 2;
        // array[index].h = array[index].h * 2;
    }
}

int main() {
    
    //Define Timers
    CPUTimer timer_unaligned_malloc;
    CPUTimer timer_unaligned_init;
    CPUTimer timer_unaligned_free;


    GPUTimer timer_unaligned_cudaMalloc;
    GPUTimer timer_unaligned_HD;
    GPUTimer timer_unaligned_kernel;
    GPUTimer timer_unaligned_DH;
    GPUTimer timer_unaligned_cudaFree;

    double unaligned_malloc_sum = 0;
    double unaligned_init_sum = 0;
    double unaligned_free_sum = 0;

    float unaligned_kernel_sum = 0;
    float unaligned_cudaMalloc_sum = 0;
    float unaligned_HD_sum = 0;
    float unaligned_DH_sum = 0;
    float unaligned_cudaFree_sum = 0;

    unsigned long long int number_of_elements = (DATA_SIZE * (1L << 30)) / 4;

    for (int j = 0; j < NUMBER_OF_TESTS; j++) {
        // Allocate host arrays

        timer_unaligned_malloc.start();

        UnalignedData* unalignedHostArray = (UnalignedData*) malloc (number_of_elements * sizeof(UnalignedData));

        timer_unaligned_malloc.end();
        unaligned_malloc_sum += timer_unaligned_malloc.elapsed();

        // Initialize arrays on the host

        timer_unaligned_init.start();

        for (unsigned long long int i = 0; i < number_of_elements; i++) {
            unalignedHostArray[i] = {1, 2, 3};//, 4, 5, 6};//, 7, 8};
        }

        timer_unaligned_init.end();
        unaligned_init_sum += timer_unaligned_init.elapsed();

        // Allocate device memory

        UnalignedData* d_unalignedArray;

        timer_unaligned_cudaMalloc.start();

        cudaMalloc((void**)&d_unalignedArray, number_of_elements * sizeof(UnalignedData));

        timer_unaligned_cudaMalloc.end();
        unaligned_cudaMalloc_sum += timer_unaligned_cudaMalloc.elapsed();

        // Copy host arrays to device

        timer_unaligned_HD.start();

        cudaMemcpy(d_unalignedArray, unalignedHostArray, number_of_elements * sizeof(UnalignedData), cudaMemcpyHostToDevice);

        timer_unaligned_HD.end();
        unaligned_HD_sum += timer_unaligned_HD.elapsed();

        // Define block and grid sizes

        int blockSize = 256;
        int numBlocks = 196608 / blockSize;

        
        // Measure time for unaligned array processing

        timer_unaligned_kernel.start();

        
        unalignedKernel<<<numBlocks, blockSize>>>(d_unalignedArray, number_of_elements);

        timer_unaligned_kernel.end();
        unaligned_kernel_sum += timer_unaligned_kernel.elapsed();

        // Copy device arrays to host

        timer_unaligned_DH.start();

        cudaMemcpy(unalignedHostArray, d_unalignedArray, number_of_elements * sizeof(UnalignedData), cudaMemcpyDeviceToHost);

        timer_unaligned_DH.end();
        unaligned_DH_sum += timer_unaligned_DH.elapsed();

        // Clean up

        timer_unaligned_cudaFree.start();

        cudaFree(d_unalignedArray);

        timer_unaligned_cudaFree.end();
        unaligned_cudaFree_sum += timer_unaligned_cudaFree.elapsed();

        timer_unaligned_free.start();

        free(unalignedHostArray);

        timer_unaligned_free.end();
        unaligned_free_sum += timer_unaligned_free.elapsed();

    }

    printf("For %ldGB elements each %dbytes and %d tests {\n", DATA_SIZE, 12, NUMBER_OF_TESTS);
    printf("Average malloc time : unaligned -> %lf\n", unaligned_malloc_sum / NUMBER_OF_TESTS);
    printf("Average cudaMalloc time : unaligned -> %lf\n", unaligned_cudaMalloc_sum / NUMBER_OF_TESTS);
    printf("Average init time : unaligned -> %lf\n", unaligned_init_sum / NUMBER_OF_TESTS);
    printf("Average HD time : unaligned -> %lf\n", unaligned_HD_sum / NUMBER_OF_TESTS);
    printf("Average kernel time : unaligned -> %lf\n", unaligned_kernel_sum / NUMBER_OF_TESTS);
    printf("Average DH time : unaligned -> %lf\n", unaligned_DH_sum / NUMBER_OF_TESTS);
    printf("Average cudaFree time : unaligned -> %lf\n", unaligned_cudaFree_sum / NUMBER_OF_TESTS);
    printf("Average free time : unaligned -> %lf\n", unaligned_free_sum / NUMBER_OF_TESTS);
    printf("}\n");

}