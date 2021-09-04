#pragma once
#include "camera_cpu.cuh"
#include "camera_gpu.cuh"
#include "image.h"
#include <cuda_runtime.h>
#include "environment.cuh"
// #include "matplotlibcpp.h"


#define NUM_INTERPOLATIONS 64
#define MAX_THREADS 1024

struct cameraDataForDtam{
  Eigen::Matrix3f T_r;
  Eigen::Vector3f T_t;
  Eigen::Vector2f cam_r_projected_on_cam_m;
  float cam_r_depth_on_camera_m;
  bool cam_r_in_front;
  cv::cuda::PtrStepSz<float3> query_proj_matrix;
};

__global__ void prepareCameraForDtam_kernel(Camera_gpu* camera_r, Camera_gpu* camera_m, cv::cuda::PtrStepSz<float3> query_proj_matrix);

__global__ void UpdateCostVolume_kernel(Camera_gpu* camera_r, Camera_gpu* camera_m, cv::cuda::PtrStepSz<int2> cost_volume,
                                                      cameraDataForDtam* camera_data_for_dtam_, float* depth_r_array);

__global__ void ComputeCostVolumeMin_kernel( cv::cuda::PtrStepSz<int2> cost_volume, float* depth_r_array);

__global__ void ComputeGradientSobelImage_kernel(cv::cuda::PtrStepSz<float> image_in, cv::cuda::PtrStepSz<float> image_out);

__global__ void ComputeDivergenceSobelImage_kernel(cv::cuda::PtrStepSz<float> image_in, cv::cuda::PtrStepSz<float> image_out);

__global__ void gradDesc_Q_toNormalize_kernel(cv::cuda::PtrStepSz<float> q, cv::cuda::PtrStepSz<float> gradient_d, float eps, float sigma_q, float* vector_to_normalize );

__global__ void gradDesc_D_kernel(cv::cuda::PtrStepSz<float> d, cv::cuda::PtrStepSz<float> a, cv::cuda::PtrStepSz<float> gradient_q, float sigma_d, float theta);

__global__ void search_A_kernel(cv::cuda::PtrStepSz<float> d, cv::cuda::PtrStepSz<float> a, cv::cuda::PtrStepSz<int2> cost_volume , float lambda, float theta, float* depth_r_array);

__global__ void sumReduction_kernel(float* v, float* v_r, int size);

__global__ void maxReduction_kernel(float* v, float* v_r, int size);

__global__ void copyArray_kernel(float* original, float* copy);

__global__ void sqrt_kernel(float* v);

__global__ void normalize_Q_kernel(float *norm, cv::cuda::PtrStepSz<float> q, float* vector_to_normalize);

__global__ void squareVectorElements_kernel(float *vector);

__global__ void Image2Vector_kernel(cv::cuda::PtrStepSz<float> image, float* vector);


class Dtam{

  public:
    CameraVector_cpu camera_vector_cpu_;
    CameraVector_gpu camera_vector_gpu_;
    int index_r_;
    float* depth_r_array_;
    cv::cuda::GpuMat cost_volume_;
    cv::cuda::GpuMat query_proj_matrix_;
    Eigen::Matrix3f T_r;
    Eigen::Vector3f T_t;
    cameraDataForDtam* camera_data_for_dtam_;


    Dtam(Environment_gpu* environment){


      int rows = environment->resolution_/environment->aspect_;
      int cols = environment->resolution_;
      float depth1_r=environment->lens_;
      float depth2_r=environment->max_depth_;
      float* depth_r_array_h = new float[NUM_INTERPOLATIONS];


      theta_end_=0.001;
      eps_=0.0001;
      beta1_=0.01;
      // beta1_=0.0001;
      // beta2_=0.05;
      // lambda_=1.0/(1.0+0.5*depth1_r);
      lambda_=0.002;
      // lambda_=0;


      for (int i=0; i<NUM_INTERPOLATIONS; i++){
        float ratio_depth_r = (float)i/((float)NUM_INTERPOLATIONS-1);
        float depth_r = depth1_r+ratio_depth_r*(depth2_r-depth1_r);
        depth_r_array_h[i]=depth_r;
      }
      cudaError_t err ;

      cudaMalloc(&depth_r_array_, sizeof(float)*NUM_INTERPOLATIONS);
      err = cudaGetLastError();
      if (err != cudaSuccess)
          printf("cudaMalloc (dtam constr) Error: %s\n", cudaGetErrorString(err));

      cudaMemcpy(depth_r_array_, depth_r_array_h, sizeof(float)*NUM_INTERPOLATIONS, cudaMemcpyHostToDevice);
      err = cudaGetLastError();
      if (err != cudaSuccess)
          printf("cudaMemcpy (dtam constr) Error: %s\n", cudaGetErrorString(err));

      delete (depth_r_array_h);

    };

    void loadCameras(CameraVector_cpu camera_vector_cpu, CameraVector_gpu camera_vector_gpu);
    void addCamera(Camera_cpu* camera_cpu, Camera_gpu* camera_gpu);
    bool setReferenceCamera(int index_r);
    void prepareCameraForDtam(int index_m);

    // void CostVolumeMin(int num_interpolations);
    // bool get1stDepthWithUV(Camera* camera_r, Camera* camera_m, Eigen::Vector2f& uv_r, Eigen::Vector2f& uv_m, float& depth);

    void updateDepthMap_parallel_gpu(int index_m);

  private:
    int n_;
    float theta_;
    float theta_end_;
    float eps_;
    float beta1_;
    float beta2_;
    float lambda_;
    float sigma_q_;
    float sigma_d_;

    void UpdateCostVolume(int index_m, cameraDataForDtam* camera_data_for_dtam_ );
    void ComputeCostVolumeMin();
    void StudyCostVolumeMin(int row, int col);
    void Regularize( cv::cuda::PtrStepSz<int2> cost_volume, float* depth_r_array);
    void ComputeGradientImage(cv::cuda::GpuMat* image_in, cv::cuda::GpuMat* image_out);
    void ComputeGradientSobelImage(cv::cuda::GpuMat* image_in, cv::cuda::GpuMat* image_out);
    void ComputeDivergenceImage(cv::cuda::GpuMat* image_in, cv::cuda::GpuMat* image_out);
    void ComputeDivergenceSobelImage(cv::cuda::GpuMat* image_in, cv::cuda::GpuMat* image_out);
    void gradDesc_Q(cv::cuda::GpuMat* q, cv::cuda::GpuMat* gradient_d );
    void gradDesc_D(cv::cuda::GpuMat* d, cv::cuda::GpuMat* a, cv::cuda::GpuMat* gradient_q );
    void search_A(cv::cuda::GpuMat* d, cv::cuda::GpuMat* a );
    void getVectorNorm(float* vector_to_normalize, float* norm, int N);
    void getVectorMax(float* vector_to_normalize, float* max, int N);
    void getImageNorm(cv::cuda::GpuMat* image, float* norm);
  private:
    void Image2Vector(cv::cuda::GpuMat* image, float* vector);




};
