#include "utils.h"

__global__
void negateImage(const uchar4* const inputImageRGBA,
                 uchar4* const outputImageRGBA,
                 int numCols) {
  const int2 matrix = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);
  const int index = matrix.y * numCols + matrix.x;

  uchar4 output = make_uchar4(255-inputImageRGBA[index].x, 255-inputImageRGBA[index].y, 255-inputImageRGBA[index].z, 255);

  outputImageRGBA[index] = output;
}

__global__
void flipX(const uchar4* const inputImageRGBA,
           uchar4* const outputImageRGBA,
           int numCols, int numRows) {
  const int2 matrix = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);
  const int index = matrix.y * numCols + matrix.x;

  const int outputIndex = (numRows * numCols - 1) - index;

  outputImageRGBA[outputIndex] = inputImageRGBA[index];
}

__global__
void flipY(const uchar4* const inputImageRGBA,
           uchar4* const outputImageRGBA,
           int numCols, int numRows) {
  const int2 matrix = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);
  const int index = matrix.y * numCols + matrix.x;

  const int2 outputMatrix = make_int2(numCols - matrix.x - 1, matrix.y);
  const int outputIndex = outputMatrix.y * numCols + outputMatrix.x;

  outputImageRGBA[outputIndex] = inputImageRGBA[index];
}


//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  const int2 matrix = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);
  const int index = matrix.y * numCols + matrix.x;
    
  if(matrix.x >= numCols || matrix.y >= numRows)
      return;
  redChannel[index]   = inputImageRGBA[index].x;
  greenChannel[index] = inputImageRGBA[index].y;
  blueChannel[index]  = inputImageRGBA[index].z;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = 255-redChannel[thread_1D_pos];
  unsigned char green = 255-greenChannel[thread_1D_pos];
  unsigned char blue  = 255-blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));


  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
  checkCudaErrors(cudaMalloc(&d_filter, sizeof( float) * filterWidth * filterWidth));


  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));

}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  //Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize(32, 32);


  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  const dim3 gridSize(numCols/blockSize.x + 1, numRows/blockSize.y + 1);


  //negateImage<<<gridSize, blockSize>>>(d_inputImageRGBA, d_outputImageRGBA, numCols);
  //flipX<<<gridSize, blockSize>>>(d_inputImageRGBA, d_outputImageRGBA, numCols, numRows);
  flipY<<<gridSize, blockSize>>>(d_inputImageRGBA, d_outputImageRGBA, numCols, numRows);

  // //Launch a kernel for separating the RGBA image into different color channels
  // separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
  //                                           numRows,
  //                                           numCols,
  //                                           d_red,
  //                                           d_green,
  //                                           d_blue);

  // // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // // launching your kernel to make sure that you didn't make any mistakes.
  // cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // // launching your kernel to make sure that you didn't make any mistakes.
  // cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // // Now we recombine your results. We take care of launching this kernel for you.
  // //
  // // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // // which you must set yourself.
  // recombineChannels<<<gridSize, blockSize>>>(d_red,
  //                                            d_green,
  //                                            d_blue,
  //                                            d_outputImageRGBA,
  //                                            numRows,
  //                                            numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}


//Free all the memory that we allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
}
