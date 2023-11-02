// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)



__global__ void add(float *input, float *output, float *output1, int len) {
  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockIdx.x*blockDim.x;
  __shared__ float num;
  num = 0;
  if(t==0 && blockIdx.x!=0){
    num = output1[blockIdx.x-1];
  }
  __syncthreads();
  if (start+t < len)
    output[start+t] = input[start+t] + num;
  if (start+t+ blockDim.x < len)
    output[start+t+ blockDim.x] = input[start+t+ blockDim.x] + num;
}




__global__ void scan1(float *input, float *output, int len) {
  
  __shared__ float T[2*BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int b = ceil((1.0*len)/(2*BLOCK_SIZE));
  T[t] = t<b ? input[(t+1)*2*blockDim.x-1] : 0;
  __syncthreads();
  int stride = 1;
  while(stride < 2*BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0)
      T[index] += T[index-stride];
    stride = stride*2;
  }
  __syncthreads();
  stride = BLOCK_SIZE/2;
  while(stride > 0) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < 2*BLOCK_SIZE) 
      T[index+stride] += T[index];
    stride = stride / 2; 
    __syncthreads();
  }
  output[t] = t<b ? T[t] : 0;
}



__global__ void scan(float *input, float *output, int len) {
  __shared__ float T[2*BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockIdx.x*blockDim.x;
  T[t] = input[start+t];
  T[blockDim.x+t] = input[start+t+blockDim.x];
  if(start + t >= len){
    T[t] = 0;
  }
  if(start + t + blockDim.x >= len){
    T[blockDim.x+t] = 0;
  }
  __syncthreads();
  int stride = 1;
  while(stride < 2*BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0)
      T[index] += T[index-stride];
    stride = stride*2;
  }
  __syncthreads();
  stride = BLOCK_SIZE/2;
  while(stride > 0) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < 2*BLOCK_SIZE) 
      T[index+stride] += T[index];
    stride = stride / 2; 
    __syncthreads();
  }
  if(start + t < len)
    output[start + t] = T[t];
  if(start + t + blockDim.x < len)
    output[start + t + blockDim.x] = T[t + blockDim.x];
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceOutput1;
  float *deviceOutput2;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput1, 2*BLOCK_SIZE * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput2, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(deviceOutput1, 0, 2*BLOCK_SIZE * sizeof(float)));
  wbCheck(cudaMemset(deviceOutput2, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil((1.0*numElements)/(2*BLOCK_SIZE)), 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(1, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements);
  cudaDeviceSynchronize();
  scan1<<<dimGrid, DimBlock>>>(deviceOutput, deviceOutput1, numElements);
  cudaDeviceSynchronize();
  add<<<DimGrid, DimBlock>>>(deviceOutput, deviceOutput2, deviceOutput1, numElements);
  cudaDeviceSynchronize();



  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput2, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceOutput1);
  cudaFree(deviceOutput2);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
