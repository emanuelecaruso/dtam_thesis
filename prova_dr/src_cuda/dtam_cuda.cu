#include "dtam_cuda.cuh"
#include <math.h>
#include "utils.h"
#include <stdlib.h>
#include "defs.h"
#include "cuda_utils.cuh"


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

  cost_volume_.create(rows,cols*NUM_INTERPOLATIONS,CV_8UC2);
  // uchar2 init_val;
  // init_val
  cost_volume_.setTo(cv::Scalar(UCHAR_MAX,0));

  return true;

}

__global__ void ComputeGradientImage_fwd_kernel(cv::cuda::PtrStepSz<float> image_in, cv::cuda::PtrStepSz<float> image_out){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int filter_idx = blockIdx.z * blockDim.z + threadIdx.z;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  __shared__ float grad_h[10][10][9]; //horizontal gradient
  __shared__ float grad_v[10][10][9]; //vertical gradient

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

  int sobel_row = filter_idx/3;
  int sobel_col = filter_idx%3;

  int current_row = row+sobel_row-1;
  int current_col = col+sobel_col-1;

  if (current_row >0 && current_col>0 && current_row<rows-1 && current_col<cols-1){
    grad_h[threadIdx.x][threadIdx.y][filter_idx]=sobel_h(sobel_row,sobel_col)*image_in(current_row,current_col);
    grad_v[threadIdx.x][threadIdx.y][filter_idx]=sobel_v(sobel_row,sobel_col)*image_in(current_row,current_col);
  }
  else{
    grad_h[threadIdx.x][threadIdx.y][filter_idx]=0;
    grad_v[threadIdx.x][threadIdx.y][filter_idx]=0;
  }

  __syncthreads();

  if (filter_idx==0){
    float value_h =0;
    float value_v =0;
    for (int i=0; i<9; i++){
      value_h+=grad_h[threadIdx.x][threadIdx.y][i];
      value_v+=grad_v[threadIdx.x][threadIdx.y][i];
    }
    value_h=abs(value_h/6.0);
    value_v=abs(value_v/6.0);
    image_out(row,col)=value_h;
    image_out(row,col+cols)=value_v;
  }


}

__global__ void ComputeGradientImage_bwd_kernel(cv::cuda::PtrStepSz<float> image_in, cv::cuda::PtrStepSz<float> image_out){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int filter_idx = blockIdx.z * blockDim.z + threadIdx.z;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  __shared__ float grad_h[10][10][9]; //horizontal gradient
  __shared__ float grad_v[10][10][9]; //vertical gradient

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

  int sobel_row = filter_idx/3;
  int sobel_col = filter_idx%3;

  int current_row = row+sobel_row-1;
  int current_col = col+sobel_col-1;

  if (current_row >-1 && current_col>-1 && current_row<rows && current_col<cols){
    grad_h[threadIdx.x][threadIdx.y][filter_idx]=-1*sobel_h(sobel_row,sobel_col)*image_in(current_row,current_col);
    grad_v[threadIdx.x][threadIdx.y][filter_idx]=-1*sobel_v(sobel_row,sobel_col)*image_in(current_row,current_col+cols);
  }
  else{
    grad_h[threadIdx.x][threadIdx.y][filter_idx]=0;
    grad_v[threadIdx.x][threadIdx.y][filter_idx]=0;
  }

  __syncthreads();

  if (filter_idx==0){
    float value_h =0;
    float value_v =0;
    for (int i=0; i<9; i++){
      value_h+=grad_h[threadIdx.x][threadIdx.y][i];
      value_v+=grad_v[threadIdx.x][threadIdx.y][i];
    }
    value_h=abs(value_h/6);
    value_v=abs(value_v/6);
    image_out(row,col)=value_h+value_v;
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



  // Kernel invocation
  dim3 threadsPerBlock( 8 , 8 , 1);
  dim3 numBlocks( rows/8, cols/8 , 1);
  prepareCameraForDtam_kernel<<<numBlocks,threadsPerBlock>>>( camera_vector_gpu_[index_r_], camera_vector_gpu_[index_m], camera_data_for_dtam_h->query_proj_matrix);
  printCudaError("Kernel preparing camera for dtam "+camera_m->name_);

  cudaMalloc(&camera_data_for_dtam_, sizeof(cameraDataForDtam));
  printCudaError("cudaMalloc (dtam constr) "+camera_m->name_);

  cudaMemcpy(camera_data_for_dtam_, camera_data_for_dtam_h, sizeof(cameraDataForDtam), cudaMemcpyHostToDevice);
  printCudaError("cudaMemcpy (dtam constr) "+camera_m->name_);

  delete camera_data_for_dtam_h;

}


__global__ void UpdateCostVolume_kernel(Camera_gpu* camera_r, Camera_gpu* camera_m,
              cv::cuda::PtrStepSz<uchar2> cost_volume, cameraDataForDtam* camera_data_for_dtam_, float* depth_r_array){


  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.z * blockDim.z + threadIdx.z;

  int cols = blockDim.y*gridDim.y;

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

  int col_ = cols*i+col;

  if (!stop){

    uchar3 clr_current = camera_m->image_rgb_(pixel_current.y(),pixel_current.x());

    // int cost_current=((clr_r.x-clr_current.x)*(clr_r.x-clr_current.x)+(clr_r.y-clr_current.y)*(clr_r.y-clr_current.y)+(clr_r.z-clr_current.z)*(clr_r.z-clr_current.z));
    uchar cost_current=(abs(clr_r.x-clr_current.x)+abs(clr_r.y-clr_current.y)+abs(clr_r.z-clr_current.z))/3;

    uchar2 cost_volume_val = cost_volume(row,col_);

    cost_volume_val.x = (cost_volume_val.x*cost_volume_val.y+cost_current)/(cost_volume_val.y+1);

    cost_volume_val.y++;

    cost_volume(row,col_) = cost_volume_val;

  }

  // __shared__ int cost_array[4][4][NUM_INTERPOLATIONS];
  // __shared__ int indx_array[4][4][NUM_INTERPOLATIONS];
  //
  // cost_array[threadIdx.x][threadIdx.y][i]=cost_volume(row,col_).x;
  // indx_array[threadIdx.x][threadIdx.y][i]=i;

  // __syncthreads();
  //
  // // -----------------------------------
  // // REDUCTION
  // // Iterate of log base 2 the block dimension
	// for (int s = 1; s < NUM_INTERPOLATIONS; s *= 2) {
	// 	// Reduce the threads performing work by half previous the previous
	// 	// iteration each cycle
	// 	if (i % (2 * s) == 0) {
  //     int min_cost = min(cost_array[threadIdx.x][threadIdx.y][i + s], cost_array[threadIdx.x][threadIdx.y][i]);
  //     if (cost_array[threadIdx.x][threadIdx.y][i] > min_cost ){
  //       indx_array[threadIdx.x][threadIdx.y][i] = indx_array[threadIdx.x][threadIdx.y][i+s];
  //       cost_array[threadIdx.x][threadIdx.y][i] = min_cost ;
  //     }
	// 	}
	// 	__syncthreads();
	// }
  // if (i == 0) {
  //   camera_r->depth_map_(row,col)=depth_r_array[indx_array[threadIdx.x][threadIdx.y][0]]/camera_r->max_depth_;
  //   if (indx_array[threadIdx.x][threadIdx.y][0]==0)
  //     camera_r->depth_map_(row,col)=1;
	// }
  // // -----------------------------------

}

__global__ void ComputeCostVolumeMin_kernel( Camera_gpu* camera_r, cv::cuda::PtrStepSz<uchar2> cost_volume, float* depth_r_array){

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.z * blockDim.z + threadIdx.z;

  int cols = blockDim.y*gridDim.y;
  int col_ = cols*i+col;


  __shared__ int cost_array[4][4][NUM_INTERPOLATIONS];
  __shared__ int indx_array[4][4][NUM_INTERPOLATIONS];

  cost_array[threadIdx.x][threadIdx.y][i]=cost_volume(row,col_).x;
  indx_array[threadIdx.x][threadIdx.y][i]=i;

  // REDUCTION
  // Iterate of log base 2 the block dimension
	for (int s = 1; s < NUM_INTERPOLATIONS; s *= 2) {
		// Reduce the threads performing work by half previous the previous
		// iteration each cycle
		if (i % (2 * s) == 0) {
      int min_cost = min(cost_array[threadIdx.x][threadIdx.y][i + s], cost_array[threadIdx.x][threadIdx.y][i]);
      if (cost_array[threadIdx.x][threadIdx.y][i] > min_cost ){
        indx_array[threadIdx.x][threadIdx.y][i] = indx_array[threadIdx.x][threadIdx.y][i+s];
        cost_array[threadIdx.x][threadIdx.y][i] = min_cost ;
      }
		}
		__syncthreads();
	}
  if (i == 0) {
    camera_r->depth_map_(row,col)=depth_r_array[indx_array[threadIdx.x][threadIdx.y][0]]/depth_r_array[NUM_INTERPOLATIONS-1];
    if (indx_array[threadIdx.x][threadIdx.y][0]==0)
      camera_r->depth_map_(row,col)=1;
	}

}


void Dtam::UpdateCostVolume(int index_m, cameraDataForDtam* camera_data_for_dtam ){

  Camera_cpu* camera_r_cpu = camera_vector_cpu_[index_r_];
  Camera_gpu* camera_r_gpu = camera_vector_gpu_[index_r_];
  int cols = camera_r_cpu->depth_map_->image_.cols;
  int rows = camera_r_cpu->depth_map_->image_.rows;

  dim3 threadsPerBlock( 4 , 4 , NUM_INTERPOLATIONS);
  dim3 numBlocks( rows/4, cols/4 , 1);
  UpdateCostVolume_kernel<<<numBlocks,threadsPerBlock>>>(camera_r_gpu, camera_vector_gpu_[index_m], cost_volume_, camera_data_for_dtam, depth_r_array_);
  printCudaError("Kernel updating cost volume");
}

void Dtam::ComputeCostVolumeMin(){
  Camera_cpu* camera_r_cpu = camera_vector_cpu_[index_r_];
  Camera_gpu* camera_r_gpu = camera_vector_gpu_[index_r_];
  int cols = camera_r_cpu->depth_map_->image_.cols;
  int rows = camera_r_cpu->depth_map_->image_.rows;

  dim3 threadsPerBlock( 4 , 4 , NUM_INTERPOLATIONS);
  dim3 numBlocks( rows/4, cols/4 , 1);
  ComputeCostVolumeMin_kernel<<<numBlocks,threadsPerBlock>>>( camera_r_gpu, cost_volume_, depth_r_array_);
  printCudaError("Kernel computing cost volume min");
}

void Dtam::ComputeGradientImage_fwd(cv::cuda::GpuMat* image_in, cv::cuda::GpuMat* image_out){



  int cols = image_in->cols;
  int rows = image_in->rows;

  image_out->create(rows,cols*2,CV_32FC1);

  dim3 threadsPerBlock( 10 , 10 , 9);
  dim3 numBlocks( rows/10, cols/10 , 1);
  ComputeGradientImage_fwd_kernel<<<numBlocks,threadsPerBlock>>>(*image_in, *image_out);
  printCudaError("Kernel computing gradient");

}

void Dtam::ComputeGradientImage_bwd(cv::cuda::GpuMat* image_in, cv::cuda::GpuMat* image_out){



  int cols = image_in->cols/2;
  int rows = image_in->rows;

  image_out->create(rows,cols,CV_32FC1);

  dim3 threadsPerBlock( 10 , 10 , 9);
  dim3 numBlocks( rows/10, cols/10 , 1);
  ComputeGradientImage_bwd_kernel<<<numBlocks,threadsPerBlock>>>(*image_in, *image_out);
  printCudaError("Kernel computing gradient");

}

__global__ void gradDesc_Q_toNormalize_kernel(cv::cuda::PtrStepSz<float> q, cv::cuda::PtrStepSz<float> gradient_d, float eps, float sigma_q, float* vector_to_normalize){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  int index = row+col*rows;
  vector_to_normalize[index]=(q(row,col)+sigma_q*gradient_d(row,col))/(1+sigma_q*eps);
  // vector_to_normalize[index]=1;

}

__global__ void gradDesc_D_kernel(cv::cuda::PtrStepSz<float> d, cv::cuda::PtrStepSz<float> a, cv::cuda::PtrStepSz<float> gradient_q, float sigma_d, float theta){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  d[row,col]=(d(row,col)+sigma_d*(gradient_q(row,col)+(1.0/theta)*a(row,col)))/(1+(sigma_d/theta));

}

__global__ void search_A_kernel(cv::cuda::PtrStepSz<float> d, cv::cuda::PtrStepSz<float> a, cv::cuda::PtrStepSz<uchar2> cost_volume , float lambda, float theta, float* depth_r_array){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.z * blockDim.z + threadIdx.z;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  int col_ = cols*i+col;

  __shared__ float cost_array[4][4][NUM_INTERPOLATIONS];
  __shared__ int indx_array[4][4][NUM_INTERPOLATIONS];

  float a_i = depth_r_array[i]/depth_r_array[NUM_INTERPOLATIONS-1];

  cost_array[threadIdx.x][threadIdx.y][i]=(1.0/(2*theta))*(d(row,col)-a_i)*(d(row,col)-a_i)+lambda*cost_volume(row,col_).x;
  indx_array[threadIdx.x][threadIdx.y][i]=i;
  __syncthreads();

  // -----------------------------------
  // REDUCTION
  // Iterate of log base 2 the block dimension
	for (int s = 1; s < NUM_INTERPOLATIONS; s *= 2) {
		// Reduce the threads performing work by half previous the previous
		// iteration each cycle
		if (i % (2 * s) == 0) {
      float min_cost = fminf(cost_array[threadIdx.x][threadIdx.y][i + s], cost_array[threadIdx.x][threadIdx.y][i]);
      if (cost_array[threadIdx.x][threadIdx.y][i] > min_cost ){
        indx_array[threadIdx.x][threadIdx.y][i] = indx_array[threadIdx.x][threadIdx.y][i+s];
        cost_array[threadIdx.x][threadIdx.y][i] = min_cost ;
      }
		}
		__syncthreads();
	}

  if (i == 0) {
    a(row,col)=depth_r_array[indx_array[threadIdx.x][threadIdx.y][0]]/depth_r_array[NUM_INTERPOLATIONS-1];
    if (indx_array[threadIdx.x][threadIdx.y][0]==0)
      a(row,col)=1;
	}
  // -----------------------------------
}

// https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/sumReduction/diverged/sumReduction.cu
__global__ void sumReduction_kernel(float *v, float *v_r, int size) {
	// Allocate shared memory
	__shared__ float partial_sum[MAX_THREADS];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
  if (tid<size)
  	partial_sum[threadIdx.x] = v[tid];
  else
    partial_sum[threadIdx.x] = 0;

	__syncthreads();

	// Iterate of log base 2 the block dimension
	for (int s = 1; s < blockDim.x; s *= 2) {
		// Reduce the threads performing work by half previous the previous
		// iteration each cycle
		if (threadIdx.x % (2 * s) == 0) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}

__global__ void normalize_Q_kernel(float norm, cv::cuda::PtrStepSz<float> q, float* vector_to_normalize){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  int index = row+col*rows;

  float denominator = fmaxf(1,norm);

  q(row,col)=vector_to_normalize[index]/denominator;

}

__global__ void squareVectorElements_kernel(float *vector){
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  vector[index]=vector[index]*vector[index];

}

__global__ void sqrt_kernel(float* v){
  v[0]=sqrt(v[0]);
}

__global__ void copyArray_kernel(float* original, float* copy){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  copy[index]=original[index];
}

void Dtam::getVectorNorm(float* vector_to_normalize, float* norm, int N){


  int GRID_SIZE = N;
  int N_THREADS = N;

  float* norm_vector_i;
  cudaMalloc(&norm_vector_i, sizeof(float)*N);
  printCudaError("cudaMalloc in norm compution 1");

  copyArray_kernel<<<N/MAX_THREADS,MAX_THREADS>>>(vector_to_normalize, norm_vector_i);
  printCudaError("Copying array for norm computation");

  squareVectorElements_kernel<<<N/MAX_THREADS,MAX_THREADS>>>(norm_vector_i);
  printCudaError("Squaring terms for computing norm");

  float* norm_vector_o;
  cudaMalloc(&norm_vector_o, sizeof(float)*GRID_SIZE);
  printCudaError("cudaMalloc in norm compution 2");

  const int TB_SIZE = MAX_THREADS;
  bool init = false;

  while (GRID_SIZE>=TB_SIZE){

    int REST = GRID_SIZE % TB_SIZE;
    GRID_SIZE = GRID_SIZE / TB_SIZE;
    if (REST > 0)
      GRID_SIZE++;
    // N_THREADS = N_THREADS;

    sumReduction_kernel<<<GRID_SIZE, TB_SIZE>>>(norm_vector_i, norm_vector_o, N_THREADS);
    printCudaError("Kernel computing sum reduction for computing norm ");

    N_THREADS = GRID_SIZE;

    if (init)
      cudaFree(norm_vector_i);
    norm_vector_i=norm_vector_o;
    init = true;

  }

  cudaMalloc(&norm_vector_o, sizeof(float));
  sumReduction_kernel<<<1, TB_SIZE>>>(norm_vector_i, norm_vector_o, N_THREADS);
  printCudaError("Kernel computing sum reduction for computing norm (final)");

  cudaDeviceSynchronize();

  sqrt_kernel<<<1, 1>>>(norm_vector_o);
  printCudaError("sqrt for norm computation");

  cudaMemcpy(norm, norm_vector_o , sizeof(float), cudaMemcpyDeviceToHost);
  printCudaError("Copying result device to host");

  cudaFree(norm_vector_i);
  cudaFree(norm_vector_o);



}

void Dtam::gradDesc_Q(cv::cuda::GpuMat* q, cv::cuda::GpuMat* gradient_d ){


  int rows = q->rows;
  int cols = q->cols;
  int N = rows*cols;
  float* vector_to_normalize;
  cudaMalloc(&vector_to_normalize, sizeof(float)*N);


  dim3 threadsPerBlock( 32 , 32 , 1);
  dim3 numBlocks( rows/32, cols/32 , 1);
  gradDesc_Q_toNormalize_kernel<<<numBlocks,threadsPerBlock>>>( *q, *gradient_d, eps_, sigma_q_, vector_to_normalize );
  printCudaError("Kernel computing next q to normalize");

  float* norm = new float;
  Dtam::getVectorNorm(vector_to_normalize, norm, N);
  std::cout << *norm << std::endl;

  normalize_Q_kernel<<<numBlocks,threadsPerBlock>>> (*norm, *q, vector_to_normalize);
  printCudaError("Kernel computing sum reduction");


}

void Dtam::gradDesc_D(cv::cuda::GpuMat* d, cv::cuda::GpuMat* a, cv::cuda::GpuMat* gradient_q ){


  int rows = d->rows;
  int cols = d->cols;

  dim3 threadsPerBlock( 32 , 32 , 1);
  dim3 numBlocks( rows/32, cols/32 , 1);
  gradDesc_D_kernel<<<numBlocks,threadsPerBlock>>>( *d, *a, *gradient_q, sigma_d_, theta_);
  printCudaError("Kernel computing next d");

}

void Dtam::search_A(cv::cuda::GpuMat* d, cv::cuda::GpuMat* a ){


  int rows = d->rows;
  int cols = d->cols;

  dim3 threadsPerBlock( 4 , 4 , NUM_INTERPOLATIONS);
  dim3 numBlocks( rows/4, cols/4 , 1);
  search_A_kernel<<<numBlocks,threadsPerBlock>>>( *d, *a, cost_volume_, lambda_ , theta_, depth_r_array_);
  printCudaError("Kernel computing search on a");

}

void Dtam::Regularize(cv::cuda::PtrStepSz<uchar2> cost_volume, float* depth_r_array){

  cv::cuda::GpuMat d = camera_vector_cpu_[index_r_]->depth_map_gpu_.clone();
  cv::cuda::GpuMat a = camera_vector_cpu_[index_r_]->depth_map_gpu_.clone();
  cv::cuda::GpuMat q;
  q.create(d.rows,d.cols*2,CV_32FC1);

  cv::cuda::GpuMat* gradient_d = new cv::cuda::GpuMat;
  cv::cuda::GpuMat* gradient_q = new cv::cuda::GpuMat;

  n_ = 0;
  theta_=0.2;
  sigma_q_=0.1;
  sigma_d_=0.1;

  while(theta_>theta_end_){

    Dtam::ComputeGradientImage_fwd( &d, gradient_d ); // compute gradient of d (n)

    Dtam::gradDesc_Q( &q, gradient_d);  // compute q (n+1)

    Dtam::ComputeGradientImage_bwd( &q, gradient_q ); // compute gradient of q (n+1)

    Dtam::gradDesc_D( &d, &a, gradient_q );  // compute d (n+1)

    Dtam::search_A( &d, &a );  // compute a (n+1)

    // upgrade steps
    sigma_d_=sigma_d_/theta_;
    sigma_q_=sigma_q_*theta_;

    // upgrade theta
    float beta = (theta_>0.001) ? beta1_ : beta2_;
    theta_ = theta_*(1-beta*n_);

    n_++;  // upgrade n
    break;
  }
  std::cout << "number of iterations n: " << n_ << std::endl;




  //**************************************************************************
  // DEBUGGGGGGGGGGG

  // //------GRADIENTS------
  // cv::cuda::GpuMat* gradient = new cv::cuda::GpuMat;
  // Dtam::ComputeGradientImage_fwd( &(camera_vector_cpu_[index_r_]->depth_map_gpu_), gradient ); // compute gradient of d (n)
  // cv::Mat_< float > test;
  // (*gradient).download(test);
  // cv::imshow("gradient test", test);
  //
  // cv::cuda::GpuMat* gradient_back = new cv::cuda::GpuMat;
  // Dtam::ComputeGradientImage_bwd( gradient ,gradient_back );
  // cv::Mat_< float > test_back;
  // (*gradient_back).download(test_back);
  // cv::imshow("gradient test back", test_back);
  //
  // //-----------------------

  // //------q,d,a------
  //
  cv::Mat_< float > first_a;
  (camera_vector_cpu_[index_r_]->depth_map_gpu_).download(first_a);
  cv::imshow("first a", first_a);

  cv::Mat_< float > first_gradient_d;
  (*gradient_d).download(first_gradient_d);
  cv::imshow("first gradient_d", first_gradient_d);

  cv::Mat_< float > first_q;
  q.download(first_q);
  cv::imshow("first q", first_q);

  cv::Mat_< float > first_gradient_q;
  (*gradient_q).download(first_gradient_q);
  cv::imshow("first gradient_q", first_gradient_q);

  cv::Mat_< float > first_d;
  d.download(first_d);
  cv::imshow("first d", first_d);

  //**************************************************************************

  delete gradient_d;
  delete gradient_q;

  camera_vector_cpu_[index_r_]->depth_map_gpu_=a;

}

void Dtam::updateDepthMap_parallel_gpu(int index_m){

  Dtam::UpdateCostVolume(index_m, camera_data_for_dtam_);

  Dtam::ComputeCostVolumeMin();

  Dtam::Regularize(cost_volume_, depth_r_array_);


}
