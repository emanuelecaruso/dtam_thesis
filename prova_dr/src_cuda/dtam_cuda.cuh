#pragma once
#include "camera_cpu.cuh"
#include "camera_gpu.cuh"
#include "image.h"
#include <cuda_runtime.h>
#include "environment.cuh"



__global__ void ComputeCostVolume_kernel(Camera_gpu* camera_r, Camera_gpu* camera_m, int num_interpolations,
        int index_r, cv::cuda::PtrStepSz<int> cost_matrix,cv::cuda::PtrStepSz<uchar> n_valid_proj_matrix);

__global__ void ComputeDepthMap(Camera_gpu* camera_r, const int num_interpolations,
        cv::cuda::PtrStepSz<int> cost_matrix,cv::cuda::PtrStepSz<uchar> n_valid_proj_matrix);

class Dtam{
  public:
    CameraVector_cpu camera_vector_cpu_;
    CameraVector_gpu camera_vector_gpu_;
    int index_r_;
    cv::cuda::GpuMat cost_matrix_;
    cv::cuda::GpuMat n_valid_proj_matrix_;

    void loadCameras(CameraVector_cpu camera_vector_cpu, CameraVector_gpu camera_vector_gpu);
    void addCamera(Camera_cpu* camera_cpu, Camera_gpu* camera_gpu);
    bool setReferenceCamera(int index_r);

    // void CostVolumeMin(int num_interpolations);
    // bool get1stDepthWithUV(Camera* camera_r, Camera* camera_m, Eigen::Vector2f& uv_r, Eigen::Vector2f& uv_m, float& depth);
    void getDepthMap(int num_interpolations, bool check=false);
};
