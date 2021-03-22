#include "dtam_cuda.cuh"
#include <math.h>
#include "utils.h"
#include <stdlib.h>

void Dtam::loadCameras(CameraVector_cpu camera_vector_cpu, CameraVector_gpu camera_vector_gpu){
  camera_vector_cpu_= camera_vector_cpu;
  camera_vector_gpu_ = camera_vector_gpu;
}

void Dtam::addCamera(Camera_cpu* camera_cpu, Camera_gpu* camera_gpu){
  camera_vector_cpu_.push_back(camera_cpu);
  camera_vector_gpu_.push_back(camera_gpu);
}

bool Dtam::setReferenceCamera(int index_r){

  int num_cameras = camera_vector_cpu_.size();

  if (index_r<0 || index_r>=num_cameras)
    return false;

  index_r_ = index_r;

  int cols = camera_vector_cpu_[index_r_]->depth_map_->image_.cols;
  int rows = camera_vector_cpu_[index_r_]->depth_map_->image_.rows;

  cost_matrix_.create(rows,cols*num_interpolations_,CV_32SC1);
  cost_matrix_.setTo(cv::Scalar::all(999999));
  n_valid_proj_matrix_.create(rows,cols*num_interpolations_,CV_8UC1);
  n_valid_proj_matrix_.setTo(cv::Scalar::all(0));

  return true;

}


__global__ void prepareCameraForDtam_kernel(Camera_gpu* camera_r, Camera_gpu* camera_m, cv::cuda::PtrStepSz<float3> query_proj_matrix){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  Eigen::Vector2i pixel_coords_r(col,row);

  // query point
  Eigen::Vector3f query_p;
  Eigen::Vector2f uv_r;
  camera_r->pixelCoords2uv(pixel_coords_r, uv_r);
  camera_r->pointAtDepth(uv_r, camera_r->max_depth_, query_p);

  // project query point
  Eigen::Vector2f query_p_projected_on_cam_m;
  float query_depth_on_camera_m;
  bool query_in_front = camera_m->projectPoint(query_p, query_p_projected_on_cam_m, query_depth_on_camera_m);

  float3 val;
  if (!query_in_front)
    val = make_float3( -1,-1,-1 );
  else
    val = make_float3( query_p_projected_on_cam_m.x(), query_p_projected_on_cam_m.y(), query_depth_on_camera_m );

  query_proj_matrix(row,col)=val;

}

void Dtam::prepareCameraForDtam(int index_m){
  Camera_cpu* camera_r = camera_vector_cpu_[index_r_];
  Eigen::Vector3f camera_r_p = camera_r->frame_camera_wrt_world_->translation();
  Camera_cpu* camera_m = camera_vector_cpu_[index_m];

  int cols = camera_r->depth_map_->image_.cols;
  int rows = camera_r->depth_map_->image_.rows;

  // project camera_r on camera_m
  Eigen::Vector2f cam_r_projected_on_cam_m;
  float cam_r_depth_on_camera_m;
  bool cam_r_in_front = camera_m->projectPoint(camera_r_p, cam_r_projected_on_cam_m, cam_r_depth_on_camera_m);

  Eigen::Isometry3f T = (*(camera_m->frame_world_wrt_camera_))*(*(camera_r->frame_camera_wrt_world_));
  Eigen::Matrix3f r=T.linear();
  Eigen::Vector3f t=T.translation();

  cameraDataForDtam* camera_data_for_dtam_h = new cameraDataForDtam;
  camera_data_for_dtam_h->T_r=r;
  camera_data_for_dtam_h->T_t=t;
  camera_data_for_dtam_h->cam_r_projected_on_cam_m=cam_r_projected_on_cam_m;
  camera_data_for_dtam_h->cam_r_depth_on_camera_m=cam_r_depth_on_camera_m;
  camera_data_for_dtam_h->cam_r_in_front=cam_r_in_front;
  query_proj_matrix_.create(rows,cols,CV_32FC3);
  camera_data_for_dtam_h->query_proj_matrix=query_proj_matrix_;

  cudaError_t err ;

  // Kernel invocation
  dim3 threadsPerBlock( 8 , 8 , 1);
  dim3 numBlocks( rows/8, cols/8 , 1);
  prepareCameraForDtam_kernel<<<numBlocks,threadsPerBlock>>>( camera_vector_gpu_[index_r_], camera_vector_gpu_[index_m], camera_data_for_dtam_h->query_proj_matrix);
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("Kernel preparing camera for dtam Error: %s\n", cudaGetErrorString(err));

  cudaDeviceSynchronize();

  cudaMalloc(&camera_data_for_dtam_, sizeof(cameraDataForDtam));
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("cudaMalloc (dtam constr) Error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(camera_data_for_dtam_, camera_data_for_dtam_h, sizeof(cameraDataForDtam), cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("cudaMemcpy (dtam constr) %s%s",camera_m->name_," Error: %s\n", cudaGetErrorString(err));

  delete camera_data_for_dtam_h;

}


__global__ void ComputeCostVolumeParallelGpu_kernel(Camera_gpu* camera_r, Camera_gpu* camera_m, int num_interpolations, cv::cuda::PtrStepSz<int> cost_matrix,
            cv::cuda::PtrStepSz<uchar> n_valid_proj_matrix, cameraDataForDtam* camera_data_for_dtam_, float* depth_r_array){


  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.z * blockDim.z + threadIdx.z;

  // initializations
  Eigen::Vector2f uv_r;
  bool stop = false;

  __shared__ float depth1_r, depth2_r;
  __shared__ Eigen::Vector2f uv1_fixed, uv2_fixed;
  __shared__ float depth1_m_fixed, depth2_m_fixed;
  __shared__ Eigen::Matrix3f r;
  __shared__ Eigen::Vector3f t;
  __shared__ float f, w, h;
  __shared__ uchar3 clr_r;

  if (i==0){
    clr_r = camera_r->image_rgb_(row,col);
    depth1_r=camera_r->lens_;
    depth2_r=camera_r->max_depth_;
    float3 val = camera_data_for_dtam_->query_proj_matrix(row,col);
    if (val.z<0)
      stop = true;
    uv1_fixed = camera_data_for_dtam_->cam_r_projected_on_cam_m;
    uv2_fixed.x()=val.x;
    uv2_fixed.y()=val.y;
    depth1_m_fixed = camera_data_for_dtam_->cam_r_depth_on_camera_m;
    depth2_m_fixed = val.z;
    r=camera_data_for_dtam_->T_r;
    t=camera_data_for_dtam_->T_t;
    f = camera_m->lens_;
    w=camera_m->width_;
    h=camera_m->width_/camera_m->aspect_;
  }

  __syncthreads();

  Eigen::Vector2i pixel_current;

  if(!stop){

    float depth_r = depth_r_array[i];

    float depth_m = depth_r*r(2,2)-t(2)-((depth_r*r(2,0)*(2*col-w))/(2*f))-((depth_r*r(2,1)*(-2*row+h))/(2*f));
    float ratio_invdepth_m = ((1.0/depth_m)-(1.0/depth1_m_fixed))/((1.0/depth2_m_fixed)-(1.0/depth1_m_fixed));

    Eigen::Vector2f uv_current;
    uv_current.x()=uv1_fixed.x()+ratio_invdepth_m*(uv2_fixed.x()-uv1_fixed.x()) ;
    uv_current.y()=uv1_fixed.y()+ratio_invdepth_m*(uv2_fixed.y()-uv1_fixed.y()) ;


    camera_m->uv2pixelCoords( uv_current, pixel_current);


    if(pixel_current.x()<0 || pixel_current.y()<0 || pixel_current.x()>=(camera_m->resolution_) || pixel_current.y()>=(int)((float)camera_m->resolution_/(float)camera_m->aspect_) )
      stop=true;
  }

  int col_ = camera_m->resolution_*i+col;

  if (!stop){

    uchar3 clr_current = camera_m->image_rgb_(pixel_current.y(),pixel_current.x());

    int cost_current=((clr_r.x-clr_current.x)*(clr_r.x-clr_current.x)+(clr_r.y-clr_current.y)*(clr_r.y-clr_current.y)+(clr_r.z-clr_current.z)*(clr_r.z-clr_current.z));
    int n_valid_proj_current = n_valid_proj_matrix(row,col_);

    if (cost_matrix(row,col_)==999999){
      cost_matrix(row,col_) = cost_current;
    }
    else{
      cost_matrix(row,col_) = (cost_matrix(row,col_)*n_valid_proj_current+cost_current)/(n_valid_proj_current+1);
    }
    n_valid_proj_matrix(row,col_)=n_valid_proj_current+1;

  }

  extern __shared__ int cost_array[];

  cost_array[i]=cost_matrix(row,col_);


  __syncthreads();

  // TODO this may be inefficient
  if (i==0){
    int min_value=999999;
    int min_index=-1;
    for (int j=0; j<num_interpolations; j++){

      if (cost_array[j]<min_value){
        min_value=cost_array[j];
        min_index=j;
      }
    }
    if (min_index==-1)
      camera_r->depth_map_(row,col)=1;
    else
      camera_r->depth_map_(row,col)=depth_r_array[min_index]/camera_r->max_depth_;
  }

}


void Dtam::updateDepthMap_parallel_gpu(int index_m){

  cudaError_t err ;

  // // reference camera
  Camera_cpu* camera_r_cpu = camera_vector_cpu_[index_r_];
  Camera_gpu* camera_r_gpu = camera_vector_gpu_[index_r_];
  int cols = camera_r_cpu->depth_map_->image_.cols;
  int rows = camera_r_cpu->depth_map_->image_.rows;

  // Kernel invocation for computing cost volume
  dim3 threadsPerBlock( 1 , 1 , num_interpolations_);
  dim3 numBlocks( rows, cols , 1);


  ComputeCostVolumeParallelGpu_kernel<<<numBlocks,threadsPerBlock,num_interpolations_*sizeof(int)>>>(camera_vector_gpu_[index_r_], camera_vector_gpu_[index_m], num_interpolations_, cost_matrix_ , n_valid_proj_matrix_, camera_data_for_dtam_, depth_r_array_);
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("Kernel computing cost volume Error: %s\n", cudaGetErrorString(err));

  cudaDeviceSynchronize();

  camera_r_cpu->depth_map_gpu_.download(camera_r_cpu->depth_map_->image_);


}
