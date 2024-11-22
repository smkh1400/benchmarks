#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp deviceProp;
    int device;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);

    printf("AsyncEngineCount is %d\n", deviceProp.asyncEngineCount);

    return 0;
}
