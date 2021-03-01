#pragma once
#include "camera_cpu.cuh"
#include "camera_gpu.cuh"
#include "image.h"
#include <cuda_runtime.h>

struct cameraData{
  Eigen::Vector2f uv1, uv2, uv1_fixed, uv2_fixed;
  float depth1_m, depth2_m, depth1_m_fixed, depth2_m_fixed;
  Eigen::Matrix3f r;
  Eigen::Vector3f t;
  Eigen::Vector2f cam_r_projected_on_cam_m;
  float cam_r_depth_on_camera_m;
  bool cam_r_in_front;
};

__global__ void CostVolumeMin_kernel(cv::cuda::PtrStepSz<uchar3> dOutput, cameraData* d_cameraData_vector_device, int n_cameras);

class Dtam{
  public:

    void CostVolumeMin(CameraVector_gpu camera_vector_gpu, cameraData* d_cameraData_vector, int n_cameras);
    // bool get1stDepthWithUV(Camera* camera_r, Camera* camera_m, Eigen::Vector2f& uv_r, Eigen::Vector2f& uv_m, float& depth);
    void getDepthMap(int num_interpolations, CameraVector_cpu& camera_vector_cpu, CameraVector_gpu& camera_vector_gpu, bool check=false);
};
