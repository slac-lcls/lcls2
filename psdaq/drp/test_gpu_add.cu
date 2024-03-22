#include <stdio.h>
#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add1(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

__global__
void add2(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

__global__
void add3(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  //printf("blockIdx %d, blockDim %d, threadIdx %d, gridDim %d, index %d, stride %d\n",
  //       blockIdx.x, blockDim.x, threadIdx.x, gridDim.x, index, stride);
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

void try1(int n, float *x, float *y)
{
  // initialize x and y arrays on the host
  for (int i = 0; i < n; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  int device = -1;
  cudaGetDevice( &device );
  cudaMemPrefetchAsync( x, n * sizeof( float ), device, 0 );
  cudaMemPrefetchAsync( y, n * sizeof( float ), device, 0 );

  // Run kernel on 1M elements on the GPU
  add1<<<1, 1>>>(n, x, y);    // 110 ms,  86 ms

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < n; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
}

void try2(int n, float *x, float *y)
{
  // initialize x and y arrays on the host
  for (int i = 0; i < n; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  int device = -1;
  cudaGetDevice( &device );
  cudaMemPrefetchAsync( x, n * sizeof( float ), device, 0 );
  cudaMemPrefetchAsync( y, n * sizeof( float ), device, 0 );

  // Run kernel on 1M elements on the GPU
  int blockSize = 256;  // 2.4 ms
  //int blockSize = 1024; // 1.6 ms  // max allowed?
  add2<<<1, blockSize>>>(n, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < n; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
}

void try3(int n, float *x, float *y)
{
  // initialize x and y arrays on the host
  for (int i = 0; i < n; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  int device = -1;
  cudaGetDevice( &device );
  cudaMemPrefetchAsync( x, n * sizeof( float ), device, 0 );
  cudaMemPrefetchAsync( y, n * sizeof( float ), device, 0 );

  // Run kernel on 1M elements on the GPU
  //int blockSize = 32; // 2.3 ms
  int blockSize = 256; // 2.2 ms, 2.7 ms
  ////int blockSize = 1024; // 2.3 ms
  int numBlocks = (n + blockSize - 1) / blockSize;
  add3<<<numBlocks, blockSize>>>(n, x, y);
  //add3<<<1024, 1024>>>(n, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < n; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  for (unsigned i = 0; i < 10; ++i)  try1(N, x, y);
  for (unsigned i = 0; i < 10; ++i)  try2(N, x, y);
  for (unsigned i = 0; i < 10; ++i)  try3(N, x, y);

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
