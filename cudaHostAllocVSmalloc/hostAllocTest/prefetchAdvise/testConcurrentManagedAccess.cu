#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp deviceProp;
    int device;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);

    printf("ConcurrentManagedAccess is %d\n", deviceProp.concurrentManagedAccess);

    return 0;
}
