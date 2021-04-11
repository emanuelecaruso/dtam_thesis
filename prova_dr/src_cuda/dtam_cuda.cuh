#pragma once
#include "camera_cpu.cuh"
#include "camera_gpu.cuh"
#include "image.h"
#include <cuda_runtime.h>
#include "environment.cuh"

struct cameraDataForDtam{
  Eigen::Matrix3f T_r;
  Eigen::Vector3f T_t;
  Eigen::Vector2f cam_r_projected_on_cam_m;
  float cam_r_depth_on_camera_m;
  bool cam_r_in_front;
  cv::cuda::PtrStepSz<float3> query_proj_matrix;
};

__global__ void prepareCameraForDtam_kernel(Camera_gpu* camera_r, Camera_gpu* camera_m, cv::cuda::PtrStepSz<float3> query_proj_matrix);

__global__ void ComputeCostVolumeParallelGpu_kernel(Camera_gpu* camera_r, Camera_gpu* camera_m, int num_interpolations,
          cv::cuda::PtrStepSz<uchar2> cost_volume, cameraDataForDtam* camera_data_for_dtam_, float* depth_r_array);

__global__ void ComputeGradientImage_kernel(cv::cuda::PtrStepSz<float> image_in, cv::cuda::PtrStepSz<float> image_out);


class Dtam{

  public:
    CameraVector_cpu camera_vector_cpu_;
    CameraVector_gpu camera_vector_gpu_;
    int index_r_;
    int num_interpolations_;
    float* depth_r_array_;
    cv::cuda::GpuMat cost_volume_;
    cv::cuda::GpuMat query_proj_matrix_;
    Eigen::Matrix3f T_r;
    Eigen::Vector3f T_t;
    cameraDataForDtam* camera_data_for_dtam_;

    Dtam(Environment* environment, int num_interpolations){
      num_interpolations_ = num_interpolations;

      int rows = environment->resolution_/environment->aspect_;
      int cols = environment->resolution_;
      float depth1_r=environment->lens_;
      float depth2_r=environment->max_depth_;
      float* depth_r_array_h = new float[num_interpolations_];

      for (int i=0; i<num_interpolations_; i++){
        float ratio_depth_r = (float)i/((float)num_interpolations_-1);
        float depth_r = depth1_r+ratio_depth_r*(depth2_r-depth1_r);
        depth_r_array_h[i]=depth_r;
      }
      cudaError_t err ;

      cudaMalloc(&depth_r_array_, sizeof(float)*num_interpolations_);
      err = cudaGetLastError();
      if (err != cudaSuccess)
          printf("cudaMalloc (dtam constr) Error: %s\n", cudaGetErrorString(err));

      cudaMemcpy(depth_r_array_, depth_r_array_h, sizeof(float)*num_interpolations_, cudaMemcpyHostToDevice);
      err = cudaGetLastError();
      if (err != cudaSuccess)
          printf("cudaMemcpy (dtam constr) Error: %s\n", cudaGetErrorString(err));

    };

    void loadCameras(CameraVector_cpu camera_vector_cpu, CameraVector_gpu camera_vector_gpu);
    void addCamera(Camera_cpu* camera_cpu, Camera_gpu* camera_gpu);
    bool setReferenceCamera(int index_r);
    void prepareCameraForDtam(int index_m);

    // void CostVolumeMin(int num_interpolations);
    // bool get1stDepthWithUV(Camera* camera_r, Camera* camera_m, Eigen::Vector2f& uv_r, Eigen::Vector2f& uv_m, float& depth);
    void getDepthMap(int num_interpolations, bool check=false);
    void updateDepthMap_parallel_gpu(int index_m);
};
