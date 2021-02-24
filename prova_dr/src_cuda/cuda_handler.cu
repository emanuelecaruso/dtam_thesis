#include "cuda_handler.cuh"
#include <stdlib.h>
#include <stdio.h>


__global__ void CostVolumeMin_kernel(cv::cuda::PtrStepSz<uchar3> dOutput, cameraData* d_cameraData_vector, int n_cameras){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  // (*d_depth_map).at<float>(0,0) = 0.0;
  // d_depth_map[0] = 0.0;
  // printf("\n");
  // printf(d_depth_map[0]);
  // printf("\n");
  // d_depth_map[0] = static_cast<unsigned char>(0.0);
  // dOutput(1, 1)=0.0;
  dOutput(row, col).x = 0;
  dOutput(row, col).y = 0;
  dOutput(row, col).z = 255;
  // a.x=1;
  // printf("\n");
  // printf(a.val[0]);
  // printf("\n");



  // struct cameraData camera_data;
  // camera_data = d_cameraData_vector[0];  //index into array




}




__host__ void CostVolumeMin(CameraVector camera_vector, cameraData* d_cameraData_vector, int n_cameras){


  // int* h_msg = (int*)malloc(sizeof(int));
  // *h_msg = -1;
  // // int msg = *h_msg;
  //
  cv::Mat_<cv::Vec3b> depth_map = camera_vector[0]->image_rgb_->image_;
  cv::cuda::GpuMat depth_map_gpu;
  depth_map_gpu.upload(depth_map);


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

  // CostVolumeMin_kernel<<<numBlocks,threadsPerBlock>>>(depth_map_gpu, d_cameraData_vector, n_cameras);
  CostVolumeMin_kernel<<<1,1>>>(depth_map_gpu, d_cameraData_vector, n_cameras);

  depth_map_gpu.download(depth_map);

  depth_map_gpu.release();
  //
  // cudaMemcpy(h_depth_map, d_depth_map, grayBytes, cudaMemcpyDeviceToHost);
  // cudaFree(d_depth_map);

  //
  // std::cout << *h_msg << std::endl;

}
