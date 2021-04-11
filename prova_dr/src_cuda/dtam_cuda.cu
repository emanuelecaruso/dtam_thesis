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

  cost_volume_.create(rows,cols*num_interpolations_,CV_8UC2);
  // uchar2 init_val;
  // init_val
  cost_volume_.setTo(cv::Scalar(UCHAR_MAX,0));

  return true;

}

__global__ void ComputeGradientImage_kernel(cv::cuda::PtrStepSz<float> image_in, cv::cuda::PtrStepSz<float> image_out){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int filter_idx = blockIdx.z * blockDim.z + threadIdx.z;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  float value_in = image_in(row,col);
  float value_out = 0;

  __shared__ float grad_h[9]; //horizontal gradient
  __shared__ float grad_v[9]; //vertical gradient

  int sobel_row = filter_idx/3;
  int sobel_col = filter_idx%3;

  //hotizontal sobel filter
  Eigen::Matrix3f sobel_h;
  sobel_h <<  +1,  0, -1,
              +2,  0, -2,
              +1,  0, -1;

  //vertical sobel filter
  Eigen::Matrix3f sobel_v;
  sobel_v <<  +1, +2, +1,
               0,  0,  0,
              -1, -2, -1;

  int current_row = row+sobel_row-1;
  int current_col = col+sobel_col-1;

  if (current_row >0 && current_col>0 && current_row<rows && current_col<cols){
    grad_h[filter_idx]=sobel_h(sobel_row,sobel_col)*image_in(current_row,current_col);
    grad_v[filter_idx]=sobel_h(sobel_row,sobel_col)*image_in(current_row,current_col);
  }
  else{
    grad_h[filter_idx]=0;
    grad_v[filter_idx]=0;
  }

  __syncthreads();

  if (filter_idx==0){
    float value_h =0;
    float value_v =0;
    float value_out;
    for (int i=0; i<9; i++){
      value_h+=grad_h[i];
      value_v+=grad_v[i];
    }
    value_h*=value_h;
    value_v*=value_v;
    value_out=sqrt(value_v+value_h);
    image_out(row,col)=value_out;
  }


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


__global__ void ComputeCostVolumeParallelGpu_kernel(Camera_gpu* camera_r, Camera_gpu* camera_m, int num_interpolations,
              cv::cuda::PtrStepSz<uchar2> cost_volume, cameraDataForDtam* camera_data_for_dtam_, float* depth_r_array){


  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.z * blockDim.z + threadIdx.z;

  // initializations
  Eigen::Vector2f uv_r;
  bool stop = false;

  uchar3 clr_r = camera_r->image_rgb_(row,col);
  float depth1_r=camera_r->lens_;
  float depth2_r=camera_r->max_depth_;
  float3 val = camera_data_for_dtam_->query_proj_matrix(row,col);
  if (val.z<0)
    stop = true;
  Eigen::Vector2f uv1_fixed = camera_data_for_dtam_->cam_r_projected_on_cam_m;
  Eigen::Vector2f uv2_fixed;
  uv2_fixed.x()=val.x;
  uv2_fixed.y()=val.y;
  float depth1_m_fixed = camera_data_for_dtam_->cam_r_depth_on_camera_m;
  float depth2_m_fixed = val.z;
  Eigen::Matrix3f r=camera_data_for_dtam_->T_r;
  Eigen::Vector3f t=camera_data_for_dtam_->T_t;
  float f = camera_m->lens_;
  float w=camera_m->width_;
  float h=camera_m->width_/camera_m->aspect_;

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

    // int cost_current=((clr_r.x-clr_current.x)*(clr_r.x-clr_current.x)+(clr_r.y-clr_current.y)*(clr_r.y-clr_current.y)+(clr_r.z-clr_current.z)*(clr_r.z-clr_current.z));
    uchar cost_current=(abs(clr_r.x-clr_current.x)+abs(clr_r.y-clr_current.y)+abs(clr_r.z-clr_current.z))/3;

    uchar2 cost_volume_val = cost_volume(row,col_);

    cost_volume_val.x = (cost_volume_val.x*cost_volume_val.y+cost_current)/(cost_volume_val.y+1);

    cost_volume_val.y++;

    cost_volume(row,col_) = cost_volume_val;

  }

  extern __shared__ int cost_array[];

  cost_array[i]=cost_volume(row,col_).x;

  __syncthreads();

  // TODO this may be inefficient
  if (i==0){
    uchar min_value=UCHAR_MAX;
    uchar min_index=num_interpolations-1;
    for (int j=0; j<num_interpolations; j++){

      if (cost_array[j]<min_value){
        min_value=cost_array[j];
        min_index=j;
      }
    }
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
  dim3 threadsPerBlock_costvol( 1 , 1 , num_interpolations_);
  dim3 numBlocks_costvol( rows, cols , 1);
  ComputeCostVolumeParallelGpu_kernel<<<numBlocks_costvol,threadsPerBlock_costvol,num_interpolations_*sizeof(int)>>>(camera_vector_gpu_[index_r_], camera_vector_gpu_[index_m], num_interpolations_, cost_volume_, camera_data_for_dtam_, depth_r_array_);
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("Kernel computing cost volume Error: %s\n", cudaGetErrorString(err));

  cudaDeviceSynchronize();

  cv::cuda::GpuMat gradient_img;
  gradient_img.create(rows,cols,CV_32FC1);

  // dim3 threadsPerBlock_gradient( 1 , 1 , 9);
  // dim3 numBlocks_gradient( rows, cols , 1);
  // ComputeGradientImage_kernel<<<numBlocks_gradient,threadsPerBlock_gradient>>>(camera_r_cpu->depth_map_gpu_, gradient_img);
  // err = cudaGetLastError();
  // if (err != cudaSuccess)
  //     printf("Kernel computing gradient Error: %s\n", cudaGetErrorString(err));

}
