// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void step1(float *df, unsigned char *duc, int imageWidth, int imageHeight, int imageChannels){
  int x = blockIdx.x* blockDim.x + threadIdx.x;
  int y = blockIdx.y* blockDim.y + threadIdx.y;
  if(x<imageWidth && y<imageHeight){
    int ii = blockIdx.z * imageWidth*imageHeight + y*imageWidth + x;
    duc[ii] = (unsigned char) ((HISTOGRAM_LENGTH-1) * df[ii]);
  }
}

__global__ void step2(unsigned char *duc, unsigned char *dg, int imageWidth, int imageHeight, int imageChannels){
  int x = blockIdx.x* blockDim.x + threadIdx.x;
  int y = blockIdx.y* blockDim.y + threadIdx.y;
  if(x<imageWidth && y<imageHeight){
    int idx = y * imageWidth + x;
    unsigned char r = duc[3*idx];
    unsigned char g = duc[3*idx + 1];
    unsigned char b = duc[3*idx + 2];
    dg[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void step3(unsigned char *dg, unsigned int *dh, int imageWidth, int imageHeight, int imageChannels){
  int x = blockIdx.x* blockDim.x + threadIdx.x;
  int y = blockIdx.y* blockDim.y + threadIdx.y;
  if(x<imageWidth && y<imageHeight){
    int ii = y*imageWidth + x;
    atomicAdd( &(dh[dg[ii]]), 1);
  }
}


__global__ void step4(unsigned int *dh, float *dcdf, int imageWidth, int imageHeight, int imageChannels){
  __shared__ unsigned int cdf[HISTOGRAM_LENGTH];
  int x = threadIdx.x;
  cdf[x] = dh[x];
  for (unsigned int stride = 1; stride <= HISTOGRAM_LENGTH / 2; stride *= 2) {
    __syncthreads();
    int idx = (x + 1) * 2 * stride - 1;
    if (idx < HISTOGRAM_LENGTH) {
      cdf[idx] += cdf[idx - stride];
    }
  }
  for (int stride = HISTOGRAM_LENGTH / 4; stride > 0; stride /= 2) {
    __syncthreads();
    int idx = (x + 1) * 2 * stride - 1;
    if (idx + stride < HISTOGRAM_LENGTH) {
      cdf[idx + stride] += cdf[idx];
    }
  }
  __syncthreads();
  dcdf[x] = cdf[x] / ((float) (imageWidth * imageHeight));
}

__global__ void step5(float *df, float *dcdf, unsigned char *duc, int imageWidth, int imageHeight, int imageChannels){
  int x = blockIdx.x* blockDim.x + threadIdx.x;
  int y = blockIdx.y* blockDim.y + threadIdx.y;
  if(x<imageWidth && y<imageHeight){
    int ii = blockIdx.z * imageWidth*imageHeight + y*imageWidth + x;
    duc[ii] = (unsigned char)min(max(255*(dcdf[duc[ii]] - dcdf[0])/(1.0 - dcdf[0]), 0.0), 255.0);
    df[ii] = (float) (duc[ii]/255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *df;
  float *dcdf;
  unsigned char *duc;
  unsigned char *dg;
  unsigned int *dh;
  const char *inputImageFile;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  cudaMalloc((void**) &duc, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **) &df, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &dg, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **) &dh, HISTOGRAM_LENGTH * sizeof(int));
  cudaMalloc((void **) &dcdf, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemset((void *) dh, 0, HISTOGRAM_LENGTH * sizeof(int));
  cudaMemset((void *) dcdf, 0, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemcpy(df, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dim3 dimBlock = dim3(32, 32, 1);
  step1<<<dimGrid, dimBlock>>>(df, duc,imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
  dimBlock = dim3(32, 32, 1);
  step2<<<dimGrid, dimBlock>>>(duc, dg, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
  dimBlock = dim3(32, 32, 1);
  step3<<<dimGrid, dimBlock>>>(dg, dh, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  dimGrid  = dim3(1, 1, 1);
  dimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);
  step4<<<dimGrid, dimBlock>>>(dh, dcdf,imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32, 32, 1);
  step5<<<dimGrid, dimBlock>>>(df, dcdf, duc,imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, df, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);


  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(df);
  return 0;
}
