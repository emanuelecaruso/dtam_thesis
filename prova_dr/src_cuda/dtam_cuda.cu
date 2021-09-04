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

  cost_volume_.create(rows,cols*NUM_INTERPOLATIONS,CV_32SC2);
  // int2 init_val;
  // init_val
  cost_volume_.setTo(cv::Scalar(INT_MAX,0));

  return true;

}

__global__ void ComputeGradientSobelImage_kernel(cv::cuda::PtrStepSz<float> image_in, cv::cuda::PtrStepSz<float> image_out){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;


  if ( row >0 && col>0 && row<rows-1 && col<cols-1){
    image_out(row,col)= -image_in(row-1,col-1)-2*image_in(row-1,col)-image_in(row-1,col+1)+image_in(row+1,col-1)+2*image_in(row+1,col)+image_in(row+1,col+1);
    image_out(row,col+cols)=-image_in(row-1,col-1)-2*image_in(row,col-1)-image_in(row+1,col-1)+image_in(row-1,col+1)+2*image_in(row,col+1)+image_in(row+1,col+1);
  }
  else
  {
    image_out(row,col)= 0;
    image_out(row,col+cols)=0;

  }

}


__global__ void ComputeDivergenceSobelImage_kernel(cv::cuda::PtrStepSz<float> image_in, cv::cuda::PtrStepSz<float> image_out){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int filter_idx = blockIdx.z * blockDim.z + threadIdx.z;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  if (row >0 && col>0 && row<rows-1 && col<cols-1){
    float value_h = -image_in(row-1,col-1)-2*image_in(row-1,col)-image_in(row-1,col+1)+image_in(row+1,col-1)+2*image_in(row+1,col)+image_in(row+1,col+1);
    float value_v = -image_in(row-1,col+cols-1)-2*image_in(row,col+cols-1)-image_in(row+1,col+cols-1)+image_in(row-1,col+cols+1)+2*image_in(row,col+cols+1)+image_in(row+1,col+cols+1);
    image_out(row,col)=value_h+value_v;
  }
  else{
    image_out(row,col)=0;
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
              cv::cuda::PtrStepSz<int2> cost_volume, cameraDataForDtam* camera_data_for_dtam_, float* depth_r_array){


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
    int cost_current=(abs(clr_r.x-clr_current.x)+abs(clr_r.y-clr_current.y)+abs(clr_r.z-clr_current.z));

    int2 cost_volume_val = cost_volume(row,col_);

    cost_volume_val.x = (cost_volume_val.x*cost_volume_val.y+cost_current)/(cost_volume_val.y+1);

    cost_volume_val.y++;

    cost_volume(row,col_) = cost_volume_val;

    // if(cost_volume(row,col_).x>255)
    // printf("%i\n", cost_volume(row,col_) );

    // if (cost_current>0.01)
    // printf("row %i col %i i %i\n",row, col, i );


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

__global__ void ComputeCostVolumeMin_kernel( Camera_gpu* camera_r, cv::cuda::PtrStepSz<int2> cost_volume, float* depth_r_array){

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
      if (cost_array[threadIdx.x][threadIdx.y][i] > min_cost){
        indx_array[threadIdx.x][threadIdx.y][i] = indx_array[threadIdx.x][threadIdx.y][i+s];
        cost_array[threadIdx.x][threadIdx.y][i] = min_cost ;
      }
		}
		__syncthreads();
	}
  if (i == 0) {

    // printf("%i\n", cost_array[threadIdx.x][threadIdx.y][0] );

    float val =(float)depth_r_array[indx_array[threadIdx.x][threadIdx.y][0]]/depth_r_array[NUM_INTERPOLATIONS-1];

    camera_r->depth_map_(row,col)=val;
    if (indx_array[threadIdx.x][threadIdx.y][0]==0)
      camera_r->depth_map_(row,col)=1.0;
	}

}


__global__ void Image2Vector_kernel(cv::cuda::PtrStepSz<float> image, float* vector){

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  int index = row+col*rows;
  vector[index]=image(row,col);

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

void Dtam::StudyCostVolumeMin(int row, int col){

  // Eigen::Vector2i pixel_coords_r(col,row);
  // cv::Vec3b clr_r;
  // camera_r->image_rgb_->evalPixel(pixel_coords_r,clr_r);
  //
  // // query point
  // Eigen::Vector3f query_p;
  // Eigen::Vector2f uv_r;
  // camera_r->pixelCoords2uv(pixel_coords_r, uv_r);
  // camera_r->pointAtDepth(uv_r, depth2_r, query_p);
  // Eigen::Vector2f query_p_projected_on_cam_m;
  // float query_depth_on_camera_m;
  // bool query_in_front = camera_m->projectPoint(query_p, query_p_projected_on_cam_m, query_depth_on_camera_m);
  //
  // // if both camera r and query point are on back of camera m return false or
  // // if camera r is in front of camera m whereas query point is on the back
  // if ((!query_in_front && !cam_r_in_front) || (!query_in_front && cam_r_in_front))
  //   continue;
  //
  // int cost_min = 999999;
  // int index_min = -1;
  // Eigen::Vector2i pixel_min;
  //
  //
  // // initializations
  // Eigen::Vector2f uv1_fixed, uv2_fixed;
  // float depth1_m_fixed, depth2_m_fixed;
  //
  // uv1_fixed=cam_r_projected_on_cam_m;
  // uv2_fixed=query_p_projected_on_cam_m;
  // depth1_m_fixed=cam_r_depth_on_camera_m;
  // depth2_m_fixed=query_depth_on_camera_m;
  //
  // Eigen::Vector2i pixel_current;
  //
  //
  // for (int i=0; i<num_interpolations_; i++){
  //
  //   float depth_r = depth_r_array_[i];
  //   float depth_m = depth_r*r(2,2)-t(2)-((depth_r*r(2,0)*(2*uv_r.x()-w))/(2*f))-((depth_r*r(2,1)*(-2*uv_r.y()+h))/(2*f));
  //   float ratio_invdepth_m = ((1.0/depth_m)-(1.0/depth1_m_fixed))/((1.0/depth2_m_fixed)-(1.0/depth1_m_fixed));
  //
  //   Eigen::Vector2f uv_current;
  //   uv_current.x()=uv1_fixed.x()+ratio_invdepth_m*(uv2_fixed.x()-uv1_fixed.x()) ;
  //   uv_current.y()=uv1_fixed.y()+ratio_invdepth_m*(uv2_fixed.y()-uv1_fixed.y()) ;
  //
  //
  //   camera_m->uv2pixelCoords( uv_current, pixel_current);
  //   cv::Vec3b clr_current;
  //   bool flag = camera_m->image_rgb_->evalPixel(pixel_current,clr_current);
  //
  //   int cost_i;
  //   int col_ = col+cols*i;
  //   int num_valid_projections;
  //
  //   if (!flag){
  //     cost_matrix_->evalPixel(row,col_,cost_i);
  //   }
  //   else{
  //
  //     cost_matrix_->evalPixel(row,col_,cost_i);
  //     n_valid_proj_matrix_->evalPixel(row,col_,num_valid_projections);
  //
  //     int cost_current = mseBetween2Colors(clr_r, clr_current);
  //     if (cost_i==999999)
  //       cost_i=cost_current;
  //     else{
  //       cost_i=(cost_i*num_valid_projections+cost_current)/(num_valid_projections+1);
  //       // cost_i=cost_current;
  //     }
  //     if (check){
  //       // std::cout  << "i " << i<< ", cost_i " << cost_i << std::endl;
  //       float ratio = (float)i/(float)num_interpolations_;
  //       cv::Vec3b magenta = cv::Vec3b(ratio*255,0,ratio*255);
  //       cv::Vec3b red = cv::Vec3b(0,0,255);
  //       cv::Vec3b blue = cv::Vec3b(255,0,0);
  //       if (i==0)
  //         camera_m->image_rgb_->setPixel(pixel_current,red);
  //       else if(i==num_interpolations_-1)
  //         camera_m->image_rgb_->setPixel(pixel_current,blue);
  //       else
  //         camera_m->image_rgb_->setPixel(pixel_current,magenta);
  //       camera_r->image_rgb_->setPixel(pixel_coords_r,red);
  //     }
  //   }
  //
  //   if (cost_i<999999){
  //
  //     cost_matrix_->setPixel(row,col_,cost_i);
  //     n_valid_proj_matrix_->setPixel(row,col_,(num_valid_projections+1));
  //
  //     if (cost_i<cost_min){
  //       index_min = i;
  //       cost_min=cost_i;
  //       pixel_min=pixel_current;
  //     }
  //   }
  //
  // }
  //
  //
  // if (index_min>=0){
  //   float depth_value = depth_r_array_[index_min]/camera_r->max_depth_;
  //   camera_r->depth_map_->setPixel(pixel_coords_r,depth_value);
  // }
  // else
  //   camera_r->depth_map_->setPixel(pixel_coords_r,1);
  //
  // if (check){
  //   cv::Vec3b green = cv::Vec3b(0,255,0);
  //   camera_m->image_rgb_->setPixel(pixel_min,green);
  //
  //   break;
  // }




  // RGBABitmapImageReference *imageReference;
  // RGBABitmapImageReference *imageReference = CreateRGBABitmapImageReference();

	// std::vector<double> xs{-2, -1, 0, 1, 2};
	// std::vector<double> ys{2, -1, -2, -1, 2};
  //
	// DrawScatterPlot(imageReference, 600, 400, &xs, &ys);

	// std::vector<double> *pngdata = ConvertToPNG(imageReference->image);
	// WriteToFile(pngdata, "example1.png");
	// DeleteImage(imageReference->image);

}

void Dtam::ComputeGradientSobelImage(cv::cuda::GpuMat* image_in, cv::cuda::GpuMat* image_out){

  int cols = image_in->cols;
  int rows = image_in->rows;

  image_out->create(rows,cols*2,CV_32FC1);

  dim3 threadsPerBlock( 32 , 32 , 1);
  dim3 numBlocks( rows/32, cols/32 , 1);
  ComputeGradientSobelImage_kernel<<<numBlocks,threadsPerBlock>>>(*image_in, *image_out);
  printCudaError("Kernel computing gradient");
}


void Dtam::ComputeDivergenceSobelImage(cv::cuda::GpuMat* image_in, cv::cuda::GpuMat* image_out){

  int cols = image_in->cols/2;
  int rows = image_in->rows;

  image_out->create(rows,cols,CV_32FC1);

  dim3 threadsPerBlock( 32 , 32 , 1);
  dim3 numBlocks( rows/32, cols/32 , 1);
  ComputeDivergenceSobelImage_kernel<<<numBlocks,threadsPerBlock>>>(*image_in, *image_out);
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

  d(row,col)=(d(row,col)+sigma_d*(gradient_q(row,col)+(1.0/theta)*a(row,col)))/(1+(sigma_d/theta));
  // d(row,col)=(d(row,col)+sigma_d*(gradient_q(row,col)+(1.0/theta)*d(row,col)))/(1+(sigma_d/theta));
  // d(row,col)=(d(row,col)+sigma_d*(gradient_q(row,col)))/(1+(sigma_d/theta));
  // d(row,col)=1;

}

__global__ void search_A_kernel(cv::cuda::PtrStepSz<float> d, cv::cuda::PtrStepSz<float> a, cv::cuda::PtrStepSz<int2> cost_volume , float lambda, float theta, float* depth_r_array){
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


__global__ void maxReduction_kernel(float *v, float *v_r, int size) {
	// Allocate shared memory
	__shared__ float partial_max[MAX_THREADS];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
  if (tid<size)
  	partial_max[threadIdx.x] = v[tid];
  else
    partial_max[threadIdx.x] = 0;

	__syncthreads();

	// Iterate of log base 2 the block dimension
	for (int s = 1; s < blockDim.x; s *= 2) {
		// Reduce the threads performing work by half previous the previous
		// iteration each cycle
		if (threadIdx.x % (2 * s) == 0) {
			partial_max[threadIdx.x] = fmaxf(abs(partial_max[threadIdx.x + s]),abs(partial_max[threadIdx.x]));
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_max[0];
	}
}

__global__ void normalize_Q_kernel(float norm, cv::cuda::PtrStepSz<float> q, float* vector_to_normalize){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  int index = row+col*rows;

  float denominator = fmaxf(1,norm);

  q(row,col)=(vector_to_normalize[index]/denominator);
  // q(row,col)=vector_to_normalize[index];

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
  printCudaError("cudaMalloc in norm computation 1");

  copyArray_kernel<<<N/MAX_THREADS,MAX_THREADS>>>(vector_to_normalize, norm_vector_i);
  printCudaError("Copying array for norm computation");

  squareVectorElements_kernel<<<N/MAX_THREADS,MAX_THREADS>>>(norm_vector_i);
  printCudaError("Squaring terms for computing norm");

  float* norm_vector_o;
  cudaMalloc(&norm_vector_o, sizeof(float)*GRID_SIZE);
  printCudaError("cudaMalloc in norm computation 2");

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

void Dtam::getVectorMax(float* vector, float* max, int N){


  int GRID_SIZE = N;
  int N_THREADS = N;

  float* max_vector_i;
  cudaMalloc(&max_vector_i, sizeof(float)*N);
  printCudaError("cudaMalloc in max computation 1");

  copyArray_kernel<<<N/MAX_THREADS,MAX_THREADS>>>(vector, max_vector_i);
  printCudaError("Copying array for max computation");

  squareVectorElements_kernel<<<N/MAX_THREADS,MAX_THREADS>>>(max_vector_i);
  printCudaError("Squaring terms for computing max");

  float* max_vector_o;
  cudaMalloc(&max_vector_o, sizeof(float)*GRID_SIZE);
  printCudaError("cudaMalloc in max computation 2");

  const int TB_SIZE = MAX_THREADS;
  bool init = false;

  while (GRID_SIZE>=TB_SIZE){

    int REST = GRID_SIZE % TB_SIZE;
    GRID_SIZE = GRID_SIZE / TB_SIZE;
    if (REST > 0)
      GRID_SIZE++;
    // N_THREADS = N_THREADS;

    maxReduction_kernel<<<GRID_SIZE, TB_SIZE>>>(max_vector_i, max_vector_o, N_THREADS);
    printCudaError("Kernel computing sum reduction for computing max ");

    N_THREADS = GRID_SIZE;

    if (init)
      cudaFree(max_vector_i);
    max_vector_i=max_vector_o;
    init = true;

  }

  cudaMalloc(&max_vector_o, sizeof(float));
  maxReduction_kernel<<<1, TB_SIZE>>>(max_vector_i, max_vector_o, N_THREADS);
  printCudaError("Kernel computing sum reduction for computing max (final)");


  cudaMemcpy(max, max_vector_o , sizeof(float), cudaMemcpyDeviceToHost);
  printCudaError("Copying result device to host");

  cudaFree(max_vector_i);
  cudaFree(max_vector_o);



}


void Dtam::Image2Vector(cv::cuda::GpuMat* image, float* vector){

  int rows = image->rows;
  int cols = image->cols;

  dim3 threadsPerBlock( 32 , 32 , 1);
  dim3 numBlocks( rows/32, cols/32 , 1);
  Image2Vector_kernel<<<numBlocks,threadsPerBlock>>>( *image, vector);
  printCudaError("Kernel converting image to vector");
}

void Dtam::getImageNorm(cv::cuda::GpuMat* image, float* norm ){

  float* vector;

  int rows = image->rows;
  int cols = image->cols;
  int N = rows*cols;
  cudaMalloc(&vector, sizeof(float)*N);

  Dtam::Image2Vector( image, vector);

  Dtam::getVectorNorm(vector, norm, N);

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

  // float* norm = new float;
  // Dtam::getVectorNorm(vector_to_normalize, norm, N);
  // std::cout << "norm is: " << *norm << std::endl;
  // normalize_Q_kernel<<<numBlocks,threadsPerBlock>>> (*norm, *q, vector_to_normalize);
  // printCudaError("Kernel computing sum reduction");

  float* max = new float;
  Dtam::getVectorMax(vector_to_normalize, max, N);
  // std::cout << "max is: " << *max << std::endl;
  normalize_Q_kernel<<<numBlocks,threadsPerBlock>>> (*max, *q, vector_to_normalize);
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

void Dtam::Regularize(cv::cuda::PtrStepSz<int2> cost_volume, float* depth_r_array){


  cv::cuda::GpuMat d = camera_vector_cpu_[index_r_]->depth_map_gpu_.clone();
  cv::cuda::GpuMat a = camera_vector_cpu_[index_r_]->depth_map_gpu_.clone();


  cv::cuda::GpuMat q;
  q.create(d.rows,d.cols*2,CV_32FC1);

  cv::cuda::GpuMat* gradient_d = new cv::cuda::GpuMat;
  cv::cuda::GpuMat* gradient_q = new cv::cuda::GpuMat;

  n_ = 0;
  theta_=0.2;

  // sigma_q_=0.1;
  // sigma_d_=0.1;

  float sigma_q0=0.1;
  float sigma_d0=0.1;
  float r=0.9;

  int resolution=camera_vector_cpu_[index_r_]->resolution_;

  // cv::cuda::GpuMat a0 = camera_vector_cpu_[index_r_]->depth_map_gpu_.clone();
  // cv::Mat_< float > a_0;
  // (a0).download(a_0);
  // cv::Mat_< float > resized_image_a_0;
  // cv::resize(a_0, resized_image_a_0, cv::Size(), 800/resolution, 800/resolution, cv::INTER_NEAREST );
  // a_0.convertTo(a_0, CV_32FC1, 255.0);
  // cv::imwrite("/home/manu/Desktop/dtam_thesis/matlab_scripts/data2.png", a_0);
  //
  // cv::Mat_< float > d_0;
  // (d).download(d_0);
  // cv::Mat_< float > resized_image_d_0;
  // cv::resize(d_0, resized_image_d_0, cv::Size(), 800/resolution, 800/resolution, cv::INTER_NEAREST );
  // cv::imshow("d_0", resized_image_d_0);

  // float* norm_d0 = new float;
  // Dtam::getImageNorm(&d, norm_d0);
  // std::cout << "\nd norm 0 is: " << *norm_d0 << std::endl;

  while(theta_>theta_end_){



    Dtam::ComputeGradientSobelImage( &d, gradient_d ); // compute gradient of d (n)

    Dtam::gradDesc_Q( &q, gradient_d);  // compute q (n+1)

    Dtam::ComputeDivergenceSobelImage( &q, gradient_q ); // compute gradient of q (n+1)

    Dtam::gradDesc_D( &d, &a, gradient_q );  // compute d (n+1)



    Dtam::search_A( &d, &a );  // compute a (n+1)

    // // // upgrade steps
    // sigma_d_=sigma_d_*theta_;
    // sigma_q_=sigma_q_/theta_;

    // upgrade theta
    // std::cout << "theta: " << theta_ <<std::endl;
    // float beta = (theta_>0.001) ? beta1_ : beta2_;
    theta_ = theta_*(1-beta1_*n_);

    n_++;  // upgrade n

    float r_pow_n=pow(r,n_);
    sigma_q_=sigma_q0/r_pow_n;
    sigma_d_=sigma_d0*r_pow_n;


    // float* norm_d = new float;
    // Dtam::getImageNorm(&d, norm_d);
    // std::cout << "\nd norm is: " << *norm_d << std::endl;
    //
    // float* norm_q = new float;
    // Dtam::getImageNorm(&q, norm_q);
    // std::cout << "q norm is: " << *norm_q << std::endl;
    //
    // float* norm_sobel_d = new float;
    // Dtam::getImageNorm(gradient_d, norm_sobel_d);
    // std::cout << "sobel d norm is: " << *norm_sobel_d << std::endl;
    //
    // std::cout << "theta is: " << theta_ << std::endl;

    cv::Mat_< float > d_1;
    d.download(d_1);
    cv::Mat_< float > resized_image_d_1;
    cv::resize(d_1, resized_image_d_1, cv::Size(), 800/resolution, 800/resolution, cv::INTER_NEAREST );
    cv::imshow("d 1", resized_image_d_1);

    cv::Mat_< float > a_1;
    a.download(a_1);
    cv::Mat_< float > resized_image_a;
    cv::resize(a_1, resized_image_a, cv::Size(), 800/resolution, 800/resolution, cv::INTER_NEAREST );
    cv::imshow("a 1", resized_image_a);

    cv::Mat_< float > d_gradient;
    (*gradient_d).download(d_gradient);
    cv::Mat_< float > resized_image_d_gradient;
    cv::resize(d_gradient, resized_image_d_gradient, cv::Size(), 800/resolution, 800/resolution, cv::INTER_NEAREST );
    cv::imshow("gradient_d", resized_image_d_gradient);
    //
    // cv::Mat_< float > q_1;
    // q.download(q_1);
    // cv::Mat_< float > resized_image_q_1;
    // cv::resize(q_1, resized_image_q_1, cv::Size(), 800/resolution, 800/resolution, cv::INTER_NEAREST );
    // cv::imshow("q_1", resized_image_q_1);
    //
    // cv::Mat_< float > q_gradient;
    // (*gradient_q).download(q_gradient);
    // cv::Mat_< float > resized_image_q_gradient;
    // cv::resize(q_gradient, resized_image_q_gradient, cv::Size(), 800/resolution, 800/resolution, cv::INTER_NEAREST );
    // cv::imshow("gradient_q", resized_image_q_gradient);

    cv::waitKey(0);

    // if(n_==100)
    //   break;


  }
  sigma_q_=sigma_q0;
  sigma_d_=sigma_d0;

  // cv::Mat_< float > opt;
  // a.download(opt);
  // cv::Mat_< float > resized_image_a;
  // cv::resize(opt, resized_image_a, cv::Size(), 800/resolution, 800/resolution, cv::INTER_NEAREST );
  // cv::imshow("opt", resized_image_a);
  // cv::waitKey(1);

  // std::cout << "number of iterations n: " << n_ << std::endl;




  //**************************************************************************
  // DEBUGGGGGGGGGGG

  // //------q,d,a------

  // cv::Mat_< float > a_1;
  // a.download(a_1);
  // cv::Mat_< float > resized_image_a;
  // cv::resize(a_1, resized_image_a, cv::Size(), 800/resolution, 800/resolution, cv::INTER_NEAREST );
  // cv::imshow("a 1", resized_image_a);
  //
  // cv::Mat_< float > d_gradient;
  // (*gradient_d).download(d_gradient);
  // cv::Mat_< float > resized_image_d_gradient;
  // cv::resize(d_gradient, resized_image_d_gradient, cv::Size(), 800/resolution, 800/resolution, cv::INTER_NEAREST );
  // cv::imshow("gradient_d", resized_image_d_gradient);
  //
  // cv::Mat_< float > q_1;
  // q.download(q_1);
  // cv::Mat_< float > resized_image_q_1;
  // cv::resize(q_1, resized_image_q_1, cv::Size(), 800/resolution, 800/resolution, cv::INTER_NEAREST );
  // cv::imshow("q_1", resized_image_q_1);
  //
  // cv::Mat_< float > q_gradient;
  // (*gradient_q).download(q_gradient);
  // cv::Mat_< float > resized_image_q_gradient;
  // cv::resize(q_gradient, resized_image_q_gradient, cv::Size(), 800/resolution, 800/resolution, cv::INTER_NEAREST );
  // cv::imshow("gradient_q", resized_image_q_gradient);

  // cv::Mat_< float > d_1;
  // d.download(d_1);
  // cv::Mat_< float > resized_image_d_1;
  // cv::resize(d_1, resized_image_d_1, cv::Size(), 800/resolution, 800/resolution, cv::INTER_NEAREST );
  // cv::imshow("d 1", resized_image_d_1);




  //**************************************************************************

  delete gradient_d;
  delete gradient_q;

  // camera_vector_cpu_[index_r_]->depth_map_gpu_= a;

}

void Dtam::updateDepthMap_parallel_gpu(int index_m){

  double t_start=getTime();  // time start for computing computation time
  double t_end=getTime();    // time end for computing computation time

  t_start=getTime();
  Dtam::UpdateCostVolume(index_m, camera_data_for_dtam_);
  t_end=getTime();
  std::cerr << "discrete cost volume computation took: " << (t_end-t_start) << " ms " << std::endl;

  t_start=getTime();
  Dtam::ComputeCostVolumeMin();
  t_end=getTime();
  std::cerr << "ComputeCostVolumeMin took: " << (t_end-t_start) << " ms " << std::endl;

  t_start=getTime();
  Dtam::Regularize(cost_volume_, depth_r_array_);
  t_end=getTime();
  std::cerr << "Regularize took: " << (t_end-t_start) << " ms " << std::endl;



}
