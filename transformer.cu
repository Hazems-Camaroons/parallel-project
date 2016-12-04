#include "utils.h"

// convert all pixels to their negative values,
// done by subtracting each channel from 255
__global__
void negateImage(const uchar4* const inputImageRGBA,
                 uchar4* const outputImageRGBA,
                 int numCols, int numRows) {
  // find the 1-d index in the 2-d image
  const int2 matrix = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);
  const int index = matrix.y * numCols + matrix.x;

  if (index < numCols * numRows){
    // apply negate algorithm
    uchar4 output = make_uchar4(255-inputImageRGBA[index].x, 255-inputImageRGBA[index].y, 255-inputImageRGBA[index].z, 255);

    outputImageRGBA[index] = output;
  }
}


// flip the image over the x-axis
// done by finding the corresponding 1-d index from the end of the image
__global__
void flipX(const uchar4* const inputImageRGBA,
           uchar4* const outputImageRGBA,
           int numCols, int numRows) {
  // find the 1-d index in the 2-d image
  const int2 matrix = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);
  const int index = matrix.y * numCols + matrix.x;

  // create an output matrix to find the correct x-value 
  const int2 outputMatrix = make_int2(matrix.x, numRows - matrix.y - 1);
  const int outputIndex = outputMatrix.y * numCols + outputMatrix.x;

  if (outputIndex < numCols * numRows)
    outputImageRGBA[outputIndex] = inputImageRGBA[index];
}


// flip the image over the y-axis
// done by finding the corresponding 1-d index from the opposite side
// of the image
__global__
void flipY(const uchar4* const inputImageRGBA,
           uchar4* const outputImageRGBA,
           int numCols, int numRows) {
  // find the 1-d index in the 2-d image
  const int2 matrix = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);
  const int index = matrix.y * numCols + matrix.x;

  // create an output matrix to find the correct x-value 
  const int2 outputMatrix = make_int2(numCols - matrix.x - 1, matrix.y);
  const int outputIndex = outputMatrix.y * numCols + outputMatrix.x;

  if (outputIndex < numCols * numRows)
    outputImageRGBA[outputIndex] = inputImageRGBA[index];
}

void transform(uchar4 * const d_inputImageRGBA, uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols, std::string transformation)
{
  //Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize(32, 32);


  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  const dim3 gridSize(numCols/blockSize.x + 1, numRows/blockSize.y + 1);

  // apply the specified transformation
  if (transformation == "negate")
    negateImage<<<gridSize, blockSize>>>(d_inputImageRGBA, d_outputImageRGBA, numCols, numRows);
  else if (transformation == "flipX")
    flipX<<<gridSize, blockSize>>>(d_inputImageRGBA, d_outputImageRGBA, numCols, numRows);
  else if (transformation == "flipY")
    flipY<<<gridSize, blockSize>>>(d_inputImageRGBA, d_outputImageRGBA, numCols, numRows);
  
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
