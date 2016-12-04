/*
    Parallel Programming Final Project
    Created By: Brian Adams, Philip Petrosino, Cody Wisniewski

    Certain parts of this were re-used from Udacity CS344 Problem Set 2, such as ImageProcessor (HW2.cpp in PS 2)
*/


#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>

#include "ImageProcessor.cpp"


/*******  DEFINED IN student_func.cu *********/

void transform(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA, uchar4* const d_outputImageRGBA,
                        const size_t numRows, const size_t numCols, std::string transformation);


/*******  Begin main *********/

int main(int argc, char **argv) {
  uchar4 *h_inputImageRGBA,  *d_inputImageRGBA;
  uchar4 *h_outputImageRGBA, *d_outputImageRGBA;

  std::string input_file;
  std::string output_file;
  std::string transformation;
  switch (argc)
  {
	case 2:
	  input_file = std::string(argv[1]);
	  output_file = "PP_output.png";
	  break;
  case 3:
    input_file = std::string(argv[1]);
    output_file = "PP_output.png";
    transformation = std::string(argv[2]);

    if (transformation != "flipY" && transformation != "flipX" && transformation != "negate"){
      std::cerr << "Must use one of the 3 transformations: flipX, flipY, negate" << std::endl;
      exit(1);
    }
    break;
	default:
      std::cerr << "Usage: ./pp input_file transformation " << std::endl;
      exit(1);
  }

  //load the image and give us our input and output pointers
  preProcess(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA, input_file);

  GpuTimer timer;
  timer.Start();
  //call the transformation code
  transform(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(), transformation);
  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  //check results and output the blurred image

  size_t numPixels = numRows()*numCols();
  //copy the output back to the host
  checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

  postProcess(output_file, h_outputImageRGBA);

  return 0;
}
