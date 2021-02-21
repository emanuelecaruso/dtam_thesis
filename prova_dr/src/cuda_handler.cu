#include "cuda_handler.cuh"
#include <stdlib.h>
#include <stdio.h>


__global__ void CostVolumeMin_kernel(CameraVector* camera_vector, int* msg){
  // *camera_vector[0]->
  *msg = 20;
}

__host__ void CostVolumeMin(CameraVector camera_vector){

  // int* h_msg = (int*)malloc(sizeof(int));
  // *h_msg = -1;
  // // int msg = *h_msg;
  //
  // int* d_msg;
  // cudaMalloc(&d_msg, sizeof(int));
  //
  // cudaMemcpy(d_msg, h_msg, sizeof(int), cudaMemcpyHostToDevice);
  // CostVolumeMin_kernel<<<1,1>>>(camera_vector, d_msg);
  // cudaMemcpy(h_msg, d_msg, sizeof(int), cudaMemcpyDeviceToHost);
  // cudaFree(d_msg);
  //
  // std::cout << *h_msg << std::endl;
  // std::cout << camera_vector[0]->depth_map_ << std::endl;
  // std::cout << camera_vector[0]->image_rgb_ << std::endl;
  // std::cout << camera_vector[0]->name_ << std::endl;
  // std::cout << camera_vector[0]->lens_ << std::endl;
  // std::cout << camera_vector[0]->aspect_ << std::endl;
  // std::cout << camera_vector[0]->width_ << std::endl;
  // std::cout << camera_vector[0]->resolution_ << std::endl;
  // std::cout << camera_vector[0]->max_depth_ << std::endl;
  std::cout << camera_vector[0]->frame_camera_wrt_world_.linear() << std::endl;
  std::cout << camera_vector[0]->frame_camera_wrt_world_.translation() << std::endl;
  std::cout << camera_vector[0]->frame_world_wrt_camera_.linear() << std::endl;
  std::cout << camera_vector[0]->frame_world_wrt_camera_.translation() << std::endl;

}
