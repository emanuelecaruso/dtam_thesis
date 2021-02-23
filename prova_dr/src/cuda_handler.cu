#include "cuda_handler.cuh"
#include <stdlib.h>
#include <stdio.h>


__global__ void CostVolumeMin_kernel(cv::cuda::PtrStepSzf dOutput){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  // (*d_depth_map).at<float>(0,0) = 0.0;
  // d_depth_map[0] = 0.0;
  // printf("\n");
  // printf(d_depth_map[0]);
  // printf("\n");
  // d_depth_map[0] = static_cast<unsigned char>(0.0);
  dOutput(row, col)=0.0;

  // printf("\n%.6f\n", v);


}




__host__ void CostVolumeMin(CameraVector camera_vector){


  // int* h_msg = (int*)malloc(sizeof(int));
  // *h_msg = -1;
  // // int msg = *h_msg;
  //
  cv::Mat_<float> depth_map = camera_vector[0]->depth_map_->image_;
  cv::cuda::GpuMat src;
  src.upload(depth_map);

  // auto size = sizeof(depth_map);
  // float* d_depth_map;
  // unsigned char* h_depth_map = depth_map.ptr();

  // cudaMalloc(&d_depth_map, size);
  // cudaMemcpy(d_depth_map, h_depth_map, size, cudaMemcpyHostToDevice);
  //
  // Kernel invocation
  const int N = 1;
  dim3 threadsPerBlock(2, 2);
  dim3 numBlocks(1, 1);

  CostVolumeMin_kernel<<<numBlocks,threadsPerBlock>>>(src);

  if (cudaSuccess != cudaGetLastError())
      std::cout << "CostVolumeMin(): gave an error" << std::endl;

  src.download(depth_map);
  //
  // cudaMemcpy(h_depth_map, d_depth_map, grayBytes, cudaMemcpyDeviceToHost);
  // cudaFree(d_depth_map);

  //
  // std::cout << *h_msg << std::endl;

}
