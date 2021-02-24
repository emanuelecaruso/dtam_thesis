#pragma once
#include "camera.h"
#include "image.h"
#include <cuda_runtime.h>

__device__ struct cameraData{
  Eigen::Vector2f uv1, uv2, uv1_fixed, uv2_fixed;
  float depth1_m, depth2_m, depth1_m_fixed, depth2_m_fixed;
  Eigen::Matrix3f r;
  Eigen::Vector3f t;
  Eigen::Vector2f cam_r_projected_on_cam_m;
  float cam_r_depth_on_camera_m;
  bool cam_r_in_front;
};


__global__ void CostVolumeMin_kernel(cv::cuda::PtrStepSz<uchar3> dOutput, cameraData* d_cameraData_vector_device, int n_cameras);
__host__ void CostVolumeMin(CameraVector camera_vector, cameraData* d_cameraData_vector_device, int n_cameras);
