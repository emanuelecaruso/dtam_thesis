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

  Dtam::Initialize();

  return true;

}



__global__ void ComputeWeightedGradientSobelImage_kernel(cv::cuda::PtrStepSz<float> image_in, cv::cuda::PtrStepSz<float> image_out, cv::cuda::PtrStepSz<float> weight_matrix){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  bool down = row>=rows-1;
  bool up = row<=0;
  bool right = col>=cols-1;
  bool left = col<=0;

  float weight=weight_matrix(row,col);
  int row_u = row+up-1;
  int row_d = row-down+1;
  int col_r = col-right+1;
  int col_l = col+left-1;

  float value_h = -image_in(row_u,col_l)-2*image_in(row_u,col)-image_in(row_u,col_r)+image_in(row_d,col_l)+2*image_in(row_d,col)+image_in(row_d,col_r);
  float value_v = -image_in(row_u,col_l)-2*image_in(row,col_l)-image_in(row_d,col_l)+image_in(row_u,col_r)+2*image_in(row,col_r)+image_in(row_d,col_r);

  float value_h_ = image_in(row_u,col)-2*image_in(row,col)+image_in(row_d,col);
  float value_v_ = image_in(row,col_l)-2*image_in(row,col);+image_in(row,col_r);

  image_out(row,col)= weight*value_h;
  image_out(row,col+cols)=weight*value_v;
  image_out(row+rows,col)= weight*value_h_;
  image_out(row+rows,col+cols)=weight*value_v_;

  // image_out(row,col)= weight*(value_h+value_h_);
  // image_out(row,col+cols)=weight*(value_v+value_v_);


}

__global__ void ComputeWeights_kernel(Camera_gpu* camera_r, cv::cuda::PtrStepSz<float> weight_matrix, float alpha, float beta){

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  auto rgb_image= camera_r->image_rgb_;

  float grad_rgbr_x=-rgb_image(row-1,col-1).x-2*rgb_image(row-1,col).x-rgb_image(row-1,col+1).x+rgb_image(row+1,col-1).x+2*rgb_image(row+1,col).x+rgb_image(row+1,col+1).x;
  float grad_rgbg_x=-rgb_image(row-1,col-1).y-2*rgb_image(row-1,col).y-rgb_image(row-1,col+1).y+rgb_image(row+1,col-1).y+2*rgb_image(row+1,col).y+rgb_image(row+1,col+1).y;
  float grad_rgbb_x=-rgb_image(row-1,col-1).z-2*rgb_image(row-1,col).z-rgb_image(row-1,col+1).z+rgb_image(row+1,col-1).z+2*rgb_image(row+1,col).z+rgb_image(row+1,col+1).z;

  float grad_rgbr_y=-rgb_image(row-1,col-1).x-2*rgb_image(row,col-1).x-rgb_image(row+1,col-1).x+rgb_image(row-1,col+1).x+2*rgb_image(row,col+1).x+rgb_image(row+1,col+1).x;
  float grad_rgbg_y=-rgb_image(row-1,col-1).y-2*rgb_image(row,col-1).y-rgb_image(row+1,col-1).y+rgb_image(row-1,col+1).y+2*rgb_image(row,col+1).y+rgb_image(row+1,col+1).y;
  float grad_rgbb_y=-rgb_image(row-1,col-1).z-2*rgb_image(row,col-1).z-rgb_image(row+1,col-1).z+rgb_image(row-1,col+1).z+2*rgb_image(row,col+1).z+rgb_image(row+1,col+1).z;

  // // L1 norm
  // float grad_rgbb_norm=abs(grad_rgbr_x)+abs(grad_rgbg_x)+abs(grad_rgbb_x)+abs(grad_rgbr_y)+abs(grad_rgbg_y)+abs(grad_rgbb_y);

  // L2 norm
  float grad_rgbb_norm=grad_rgbr_x*grad_rgbr_x+grad_rgbg_x*grad_rgbg_x+grad_rgbb_x*grad_rgbb_x+grad_rgbr_y*grad_rgbr_y+grad_rgbg_y*grad_rgbg_y+grad_rgbb_y*grad_rgbb_y;
//
  float weight=exp(-alpha*pow( grad_rgbb_norm, 0.8));
  // float weight=exp(-alpha* grad_rgbb_norm);
  // float weight=1;

  weight_matrix(row,col)=weight;
  // printf("%f\n", weight );

}


__global__ void ComputeWeightedDivergenceSobelImage_kernel(cv::cuda::PtrStepSz<float> image_in, cv::cuda::PtrStepSz<float> image_out, cv::cuda::PtrStepSz<float> weight_matrix){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int filter_idx = blockIdx.z * blockDim.z + threadIdx.z;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  bool down = row>=rows-1;
  bool up = row<=0;
  bool right = col>=cols-1;
  bool left = col<=0;

  float weight=weight_matrix(row,col);
  int row_u = row+up-1;
  int row_d = row-down+1;
  int col_r = col-right+1;
  int col_l = col+left-1;

  float value_h = (-image_in(row_u,col_l)-2*image_in(row_u,col)-image_in(row_u,col_r)+image_in(row_d,col_l)+2*image_in(row_d,col)+image_in(row_d,col_r));
  float value_v = (-image_in(row_u,col_l+cols)-2*image_in(row,col_l+cols)-image_in(row_d,col_l+cols)+image_in(row_u,col_r+cols)+2*image_in(row,col_r+cols)+image_in(row_d,col_r+cols));

  float value_h_ = image_in(rows+row_u,col)-2*image_in(rows+row,col)+image_in(rows+row_d,col);
  float value_v_ = image_in(rows+row,col_l+cols)-2*image_in(row+rows,col+cols)+image_in(rows+row,col_r+cols);
  float a=1;
  float b=1;
  image_out(row,col)=weight*(a*(value_h+value_v)+b*(-value_h_-value_v_));
  // image_out(row,col)=weight*(value_h+value_v);
  // image_out(row,col)=-weight*(value_h_+value_v_);

  // image_out(row,col)=weight*(value_h-value_h_+value_v-value_v_);

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

  int cols = camera_r->invdepth_map_->image_.cols;
  int rows = camera_r->invdepth_map_->image_.rows;

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
              cv::cuda::PtrStepSz<int2> cost_volume, cameraDataForDtam* camera_data_for_dtam_,
              float* invdepth_r_array, int threshold, bool occl){


  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.z * blockDim.z + threadIdx.z;

  int cols = blockDim.y*gridDim.y;

  // initializations
  Eigen::Vector2f uv_r;
  Eigen::Vector2i pixel_coords_r(col,row);
  camera_r->pixelCoords2uv(pixel_coords_r, uv_r);
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

    float depth_r = 1.0/invdepth_r_array[i];

    float depth_m = depth_r*r(2,2)-t(2)-((depth_r*r(2,0)*(2*uv_r.x()-w))/(2*f))-((depth_r*r(2,1)*(-2*uv_r.y()+h))/(2*f));
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

    if(occl){
      cost_current=min(cost_current, 20);
      // cost_current=max(cost_current, 3);
    }

    if (cost_current<20 || occl)
    {
        // cost_current=max(cost_current, 3);
        // cost_current=min(cost_current, 50);

        int2 cost_volume_val = cost_volume(row,col_);

        cost_volume_val.x = (cost_volume_val.x*cost_volume_val.y+cost_current)/(cost_volume_val.y+1);

        cost_volume_val.y++;

        cost_volume(row,col_) = cost_volume_val;
    }

  }

}



__global__ void StudyCostVolumeMin_kernel(Camera_gpu* camera_r, Camera_gpu* camera_m,
              cv::cuda::PtrStepSz<int2> cost_volume, cameraDataForDtam* camera_data_for_dtam_, float* invdepth_r_array,
              int row, int col, cv::cuda::PtrStepSz<float> depth_groundtruth, cv::cuda::PtrStepSz<float> a){


  int i = blockIdx.z * blockDim.z + threadIdx.z;

  int cols = camera_r->resolution_;

  // initializations
  Eigen::Vector2f uv_r;
  Eigen::Vector2i pixel_coords_r(col,row);
  camera_r->pixelCoords2uv(pixel_coords_r, uv_r);

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

    float depth_r = 1.0/invdepth_r_array[i];

    // Eigen::Vector3f p;
    // float depth_m;
    // camera_r->pointAtDepth(uv_r, depth_r, p);
    // Eigen::Vector2f uv_current;
    // bool query_in_front = camera_m->projectPoint(p, uv_current, depth_m);

    float depth_m = depth_r*r(2,2)-t(2)-((depth_r*r(2,0)*(2*uv_r.x()-w))/(2*f))-((depth_r*r(2,1)*(-2*uv_r.y()+h))/(2*f));
    float ratio_invdepth_m = ((1.0/depth_m)-(1.0/depth1_m_fixed))/((1.0/depth2_m_fixed)-(1.0/depth1_m_fixed));
    // float ratio_invdepth_m = ((1.0/depth_m)-(1.0/depth1_m_fixed))/((1.0/depth2_m_fixed)-(1.0/depth1_m_fixed));

    // printf("%f\n", depth_m );

    Eigen::Vector2f uv_current;
    uv_current.x()=uv1_fixed.x()+ratio_invdepth_m*(uv2_fixed.x()-uv1_fixed.x()) ;
    uv_current.y()=uv1_fixed.y()+ratio_invdepth_m*(uv2_fixed.y()-uv1_fixed.y()) ;

    camera_m->uv2pixelCoords( uv_current, pixel_current);

    if(pixel_current.x()<0 || pixel_current.y()<0 || pixel_current.x()>=(camera_m->resolution_) || pixel_current.y()>=(int)((float)camera_m->resolution_/(float)camera_m->aspect_) )
      stop=true;
  }

  int col_ = cols*i+col;

  int cost_current;
  uchar3 clr_current;

  if (!stop){

    clr_current = camera_m->image_rgb_(pixel_current.y(),pixel_current.x());

    // int cost_current=((clr_r.x-clr_current.x)*(clr_r.x-clr_current.x)+(clr_r.y-clr_current.y)*(clr_r.y-clr_current.y)+(clr_r.z-clr_current.z)*(clr_r.z-clr_current.z));
    cost_current=(abs(clr_r.x-clr_current.x)+abs(clr_r.y-clr_current.y)+abs(clr_r.z-clr_current.z));

    int val =255-(cost_current/3);
    uchar3 magenta = make_uchar3(255,0,255);
    uchar3 magenta_ = make_uchar3(val,0,val);

    camera_m->image_rgb_(pixel_current.y(),pixel_current.x())=magenta;

  }

  __syncthreads();

  __shared__ int cost_array[NUM_INTERPOLATIONS];
  __shared__ int indx_array[NUM_INTERPOLATIONS];

  cost_array[i]=cost_volume(row,col_).x;
  indx_array[i]=i;

  // REDUCTION
  // Iterate of log base 2 the block dimension
  for (int s = 1; s < NUM_INTERPOLATIONS; s *= 2) {
    // Reduce the threads performing work by half previous the previous
    // iteration each cycle
    if (i % (2 * s) == 0) {
      int min_cost = min(cost_array[i + s], cost_array[i]);

      if (cost_array[i] > min_cost){
        indx_array[i] = indx_array[i+s];
        cost_array[i] = min_cost ;
      }
    }
    __syncthreads();
  }
  __syncthreads();
  // printf("cost current at i=%i is: %i\n",i, cost_current);

  if (i == indx_array[0]) {
  // if (i == 57) {
  // if (i == 1) {
    uchar3 blue = make_uchar3(255,0,0);
    camera_m->image_rgb_(pixel_current.y(),pixel_current.x())=blue;

    // printf("camera_r->max_depth_: %f\n", camera_r->max_depth_);
    // printf("depth1_m_fixed: %f\n", depth1_m_fixed);
    // printf("depth2_m_fixed: %f\n", depth2_m_fixed);
    printf("predicted depth from costvol: %f\n", 1.0/invdepth_r_array[indx_array[0]]);
    printf("predicted depth from a: %f\n", 1.0/(a(row,col)*invdepth_r_array[0]));
    printf("grondtruth depth: %f\n", (1.0/(depth_groundtruth(row,col)*2)));
    // printf("grondtruth val: %f\n", depth_groundtruth(row,col));
    printf("clr_r is: %i,%i,%i\n", clr_r.x ,clr_r.y ,clr_r.z);
    printf("clr_current is: %i,%i,%i\n", clr_current.x ,clr_current.y ,clr_current.z);
    // printf("stop flag is: %i\n", stop);
    printf("cost current is: %i\n", cost_current);
    printf("cost current TOT is: %i\n", cost_volume(row,col_).x);
    printf("n projections: %i\n", cost_volume(row,col_).y);
    // printf("cost current is: %i\n", cost_current);
    printf("min cost is: %i\n", cost_array[0]);
    // printf("coords: %i %i\n", pixel_current.y(), pixel_current.x());
    printf("idx: %i\n", i);
    printf("\n");
  }

}


__global__ void ComputeCostVolumeMin_kernel( Camera_gpu* camera_r, cv::cuda::PtrStepSz<int2> cost_volume, float* invdepth_r_array){

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
  if (i == indx_array[threadIdx.x][threadIdx.y][0]) {

    // printf("%i\n", cost_array[threadIdx.x][threadIdx.y][0] );

    float val =(float)invdepth_r_array[i]/invdepth_r_array[0];

    camera_r->invdepth_map_(row,col)=val;
    if (indx_array[threadIdx.x][threadIdx.y][0]==0)
      camera_r->invdepth_map_(row,col)=1.0;

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

__global__ void UpdateState_kernel(cv::cuda::PtrStepSz<float> points_added, cv::cuda::PtrStepSz<int2> cost_volume, cv::cuda::PtrStepSz<float> a, cv::cuda::PtrStepSz<float> d, cv::cuda::PtrStepSz<float> gradient_q, float* invdepth_r_array, int switch_idx, float switch_depth, float depth1_r, float depth2_r){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  if(abs(gradient_q(row,col))<0.05){

    float invdepth=a(row,col);
    float depth=depth1_r/invdepth;
    int i;
    if(depth<switch_depth){
      i= (int)roundf( switch_idx*((depth -depth1_r)/(switch_depth-depth1_r)));
      // if(depth<0.77)
      //   printf("%i,%i %f \n",row,col,depth );
    }
    else
      i=switch_idx+(int)roundf(((NUM_INTERPOLATIONS-switch_idx-1)*((1.0/depth)-(1.0/switch_depth)))/((1.0/depth2_r)-(1.0/switch_depth)));
    int col_ = cols*i+col;
    int cost = cost_volume(row,col_).x;

    //
    //
    // if(cost_volume(row,col_).x<3 && cost_volume(row,col_).y>2){
    //   points_added(row,col)=1;
    // }
    if(cost<5)
      points_added(row,col)=d(row,col);
  }

}

void Dtam::Initialize(){

  threshold_=50;
  theta_=0.07;
  // theta_switch_=0.19;
  theta_switch_=0;
  theta_end_=0.00001;
  eps_=0.1;
  alpha_=0.00002;
  // alpha_=0.00007;
  // alpha_=0;
  beta1_=0.0001;
  // beta1_=0.0002;
  beta2_=0.03;
  // lambda_=0.005;
  lambda_=0.005;
  sigma_q0_=0.1;
  sigma_d0_=0.5;
  // sigma_d0_=0.1;
  // r1_=1;
  r1_=0.87;
  // r2_=0.5;
  r2_=0.7;


  // theta_=0.7;
  // theta_switch_=0.57;
  // theta_end_=0.001;
  // eps_=0.1;
  // alpha_=0.00002;
  // beta1_=0.0002;
  // beta2_=0.02;
  // // lambda_=0.005;
  // lambda_=0.001;
  // sigma_q0_=0.1;
  // sigma_d0_=0.5;
  // r1_=0.85;
  // r2_=0.7;

  n_ = 0;
  sigma_q_=sigma_q0_;
  sigma_d_=sigma_d0_;




  int cols = camera_vector_cpu_[index_r_]->invdepth_map_->image_.cols;
  int rows = camera_vector_cpu_[index_r_]->invdepth_map_->image_.rows;

  // depth_groundtruth_.create(rows,cols,CV_32FC1);
  depth_groundtruth_ = camera_vector_cpu_[index_r_]->invdepth_map_gpu_.clone();
  camera_vector_cpu_[index_r_]->invdepth_map_gpu_.setTo(1);
  camera_vector_cpu_[index_r_]->invdepth_map_->setAllPixels(1);


  cost_volume_.create(rows,cols*NUM_INTERPOLATIONS,CV_32SC2);
  cost_volume_.setTo(cv::Scalar(INT_MAX,0));

  weight_matrix_.create(rows,cols,CV_32FC1);

  points_added_.create(rows,cols,CV_32FC1);
  points_added_.setTo(0);

  Dtam::ComputeWeights();


}

void Dtam::UpdateCostVolume(int index_m ){
  double t_s=getTime();

  Camera_cpu* camera_r_cpu = camera_vector_cpu_[index_r_];
  Camera_gpu* camera_r_gpu = camera_vector_gpu_[index_r_];
  int cols = camera_r_cpu->invdepth_map_->image_.cols;
  int rows = camera_r_cpu->invdepth_map_->image_.rows;

  dim3 threadsPerBlock( 4 , 4 , NUM_INTERPOLATIONS);
  dim3 numBlocks( rows/4, cols/4 , 1);
  UpdateCostVolume_kernel<<<numBlocks,threadsPerBlock>>>(camera_r_gpu, camera_vector_gpu_[index_m], cost_volume_, camera_data_for_dtam_, invdepth_r_array_, threshold_, frames_computed_<5);
  printCudaError("Kernel updating cost volume");

  double t_e=getTime();
  double delta=t_e-t_s;

  std::cerr << "cost volume computation took: " << delta << " ms " << std::endl;




}

void Dtam::StudyCostVolumeMin(int index_m, cameraDataForDtam* camera_data_for_dtam, int row, int col,bool showbaseline=false){

  Camera_cpu* camera_r_cpu = camera_vector_cpu_[index_r_];
  Camera_gpu* camera_r_gpu = camera_vector_gpu_[index_r_];

  dim3 threadsPerBlock( 1 , 1 , NUM_INTERPOLATIONS);
  dim3 numBlocks( 1, 1 , 1);
  StudyCostVolumeMin_kernel<<<numBlocks,threadsPerBlock>>>(camera_r_gpu, camera_vector_gpu_[index_m], cost_volume_, camera_data_for_dtam, invdepth_r_array_,row,col, depth_groundtruth_, a);
  printCudaError("Kernel studying cost volume");

  if(showbaseline){
    Image< cv::Vec3b >* study_baseline = new Image< cv::Vec3b >("Study baseline");
    camera_vector_cpu_[index_m]->image_rgb_gpu_.download(study_baseline->image_);
    study_baseline->show(1500/camera_vector_cpu_[index_m]->resolution_);
    delete study_baseline;

  }



  if(showbaseline){
    Image< cv::Vec3b >* study_ref = new Image< cv::Vec3b >("Study reference");
    camera_vector_cpu_[index_r_]->image_rgb_gpu_.download(study_ref->image_);
    Eigen::Vector2i pxl(col,row);
    cv::Vec3b red(0,0,255);
    study_ref->setPixel(pxl, red);
    study_ref->show(1500/camera_vector_cpu_[index_r_]->resolution_);
  }


}

void Dtam::ComputeCostVolumeMin(){

  double t_s=getTime();

  Camera_cpu* camera_r_cpu = camera_vector_cpu_[index_r_];
  Camera_gpu* camera_r_gpu = camera_vector_gpu_[index_r_];
  int cols = camera_r_cpu->invdepth_map_->image_.cols;
  int rows = camera_r_cpu->invdepth_map_->image_.rows;

  dim3 threadsPerBlock( 4 , 4 , NUM_INTERPOLATIONS);
  dim3 numBlocks( rows/4, cols/4 , 1);
  ComputeCostVolumeMin_kernel<<<numBlocks,threadsPerBlock>>>( camera_r_gpu, cost_volume_, invdepth_r_array_);
  printCudaError("Kernel computing cost volume min");

  double t_e=getTime();
  double delta=t_e-t_s;

  std::cerr << "ComputeCostVolumeMin took: " << delta << " ms " << std::endl;




}



void Dtam::ComputeWeightedGradientSobelImage(cv::cuda::GpuMat* image_in, cv::cuda::GpuMat* image_out){

  int cols = image_in->cols;
  int rows = image_in->rows;

  dim3 threadsPerBlock( 32 , 32 , 1);
  dim3 numBlocks( rows/32, cols/32 , 1);
  ComputeWeightedGradientSobelImage_kernel<<<numBlocks,threadsPerBlock>>>(*image_in, *image_out, weight_matrix_);
  printCudaError("Kernel computing gradient");
}


void Dtam::ComputeWeightedDivergenceSobelImage(cv::cuda::GpuMat* image_in, cv::cuda::GpuMat* image_out){

  int cols = image_in->cols/2;
  int rows = image_in->rows/2;

  dim3 threadsPerBlock( 32 , 32 , 1);
  dim3 numBlocks( rows/32, cols/32 , 1);
  ComputeWeightedDivergenceSobelImage_kernel<<<numBlocks,threadsPerBlock>>>(*image_in, *image_out, weight_matrix_);
  printCudaError("Kernel computing gradient");
}

__global__ void gradDesc_Q_toNormalize_kernel(cv::cuda::PtrStepSz<float> q, cv::cuda::PtrStepSz<float> gradient_d, float eps, float sigma_q, float* vector_to_normalize){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  int index = row+col*rows;
  vector_to_normalize[index]=(q(row,col)+sigma_q*gradient_d(row,col))/(1+sigma_q*eps);
  // vector_to_normalize[index]=(sigma_q*gradient_d(row,col))/(1+sigma_q*eps);
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

__global__ void search_A_kernel(cv::cuda::PtrStepSz<float> d, cv::cuda::PtrStepSz<float> a, cv::cuda::PtrStepSz<int2> cost_volume , cv::cuda::PtrStepSz<float> points_added, float lambda, float theta, float* invdepth_r_array){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.z * blockDim.z + threadIdx.z;

  int rows = blockDim.x*gridDim.x;
  int cols = blockDim.y*gridDim.y;

  int col_ = cols*i+col;

  __shared__ float cost_array[4][4][NUM_INTERPOLATIONS];
  __shared__ int indx_array[4][4][NUM_INTERPOLATIONS];

  float a_i = invdepth_r_array[i]/invdepth_r_array[0];

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

  if (i == indx_array[threadIdx.x][threadIdx.y][0]) {

    // else{
    //   a(row,col)=0.025;
    // }
    a(row,col)=a_i;
    if (cost_volume(row,col_).y==0)
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
	}    // void CostVolumeMin(int num_interpolations);
    // bool get1stDepthWithUV(Camera* camera_r, Camera* camera_m, Eigen::Vector2f& uv_r, Eigen::Vector2f& uv_m, float& depth);

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

  // q(row,col)=(vector_to_normalize[index]/denominator);
  q(row,col)=vector_to_normalize[index];

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


  const int TB_SIZE = MAX_THREADS;
  bool init = false;

  while (GRID_SIZE>=TB_SIZE){

    cudaMalloc(&norm_vector_o, sizeof(float)*GRID_SIZE);
    printCudaError("cudaMalloc in norm computation 2");

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

  float* max_vector_o;


  const int TB_SIZE = MAX_THREADS;
  bool init = false;

  while (GRID_SIZE>=TB_SIZE){

    cudaMalloc(&max_vector_o, sizeof(float)*GRID_SIZE);
    printCudaError("cudaMalloc in max computation 2");

    int REST = GRID_SIZE % TB_SIZE;
    GRID_SIZE = GRID_SIZE / TB_SIZE;
    if (REST > 0)
      GRID_SIZE++;
    // N_THREADS = N_THREADS;

    maxReduction_kernel<<<GRID_SIZE, TB_SIZE>>>(max_vector_i, max_vector_o, N_THREADS);
    printCudaError("Kernel computing sum reduction for computing max ");

    N_THREADS = GRID_SIZE;

    // if (init)
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

  // float* max = new float;
  // Dtam::getVectorMax(vector_to_normalize, max, N);
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
  search_A_kernel<<<numBlocks,threadsPerBlock>>>( *d, *a, cost_volume_, points_added_, lambda_ , theta_, invdepth_r_array_);
  printCudaError("Kernel computing search on a");

}

void Dtam::ComputeWeights(){

  Camera_cpu* camera_r_cpu = camera_vector_cpu_[index_r_];
  Camera_gpu* camera_r_gpu = camera_vector_gpu_[index_r_];
  int cols = camera_r_cpu->invdepth_map_->image_.cols;
  int rows = camera_r_cpu->invdepth_map_->image_.rows;
  dim3 threadsPerBlock( 32 , 32 , 1);
  dim3 numBlocks( rows/32, cols/32 , 1);
  ComputeWeights_kernel<<<numBlocks,threadsPerBlock>>>( camera_r_gpu, weight_matrix_, alpha_, beta1_);
  printCudaError("Kernel computing cost volume min");

}

void Dtam::UpdateParametersReg(bool print){


  // upgrade theta
  // std::cout << "theta: " << theta_ <<std::endl;
  float beta = (theta_>theta_switch_) ? beta1_ : beta2_;
  float r = (theta_>theta_switch_) ? r1_ : r2_;

  n_++;  // upgrade n

  theta_ = theta_*(1-beta*n_);
  float r_pow_n=pow(r,n_);
  sigma_q_=sigma_q0_/r_pow_n;
  sigma_d_=sigma_d0_*r_pow_n;

  if(print){
    std::cout << "theta_: " << theta_ << std::endl;
    std::cout << "sigma_q_: " << sigma_q_ << std::endl;
    std::cout << "sigma_d_: " << sigma_d_ << std::endl;
    std::cout << "n_: " << n_ << std::endl;
    std::cout << "beta: " << beta << std::endl;
    std::cout << "r: " << r << std::endl;
  }

}
void Dtam::showImgs(int scale){
  int resolution=camera_vector_cpu_[index_r_]->resolution_;

  cv::Mat_< float > ad_comp;


  cv::Mat_< float > d_1;
  d.download(d_1);
  cv::Mat_< float > resized_image_d_1;
  cv::resize(d_1, resized_image_d_1, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  cv::Mat_< float > a_1;
  a.download(a_1);
  cv::Mat_< float > resized_image_a;
  cv::resize(a_1, resized_image_a, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  cv::hconcat(resized_image_d_1,resized_image_a,ad_comp);

  cv::Mat_< float > gt_gterr_comp;

  cv::Mat_< float > gt;
  depth_groundtruth_.download(gt);
  cv::Mat_< float > resized_image_gt;
  cv::resize(gt, resized_image_gt, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  cv::Mat_< float > gterr;
  cv::absdiff(resized_image_gt, resized_image_d_1, gterr);

  cv::hconcat(resized_image_gt,gterr,gt_gterr_comp);

  cv::Mat_< float > out;
  cv::vconcat(ad_comp,gt_gterr_comp,out);

  cv::imshow("comparison", out);

  // Image< float >* invdepth_groundtruth = new Image< float >("invdepth groundtruth");
  // depth_groundtruth_.download(invdepth_groundtruth->image_);
  // invdepth_groundtruth->show(scale/camera_vector_cpu_[index_m]->resolution_);

  // cv::Mat_< float > d_gradient;
  // (gradient_d).download(d_gradient);
  // cv::Mat_< float > resized_image_d_gradient;
  // cv::resize(d_gradient, resized_image_d_gradient, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  // cv::imshow("gradient_d", resized_image_d_gradient);

  // cv::Mat_< float > q_1;
  // q.download(q_1);
  // cv::Mat_< float > resized_image_q_1;
  // cv::resize(q_1, resized_image_q_1, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  // cv::imshow("q_1", resized_image_q_1);
  //
  cv::Mat_< float > q_gradient;
  (gradient_q).download(q_gradient);
  cv::Mat_< float > resized_image_q_gradient;
  cv::resize(q_gradient, resized_image_q_gradient, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  cv::imshow("gradient_q", resized_image_q_gradient);

  cv::Mat_< float > pa_1;
  points_added_.download(pa_1);
  cv::Mat_< float > resized_image_pa_1;
  cv::resize(pa_1, resized_image_pa_1, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  cv::imshow("pa 1", resized_image_pa_1);

  // cv::waitKey(0);

}

void Dtam::Regularize(){

  double t_s=getTime();


  int resolution=camera_vector_cpu_[index_r_]->resolution_;

  // cv::cuda::GpuMat a0 = camera_vector_cpu_[index_r_]->invinvdepth_map_gpu_.clone();
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


  Dtam::ComputeWeightedGradientSobelImage( &d, &gradient_d ); // compute gradient of d (n)

  Dtam::gradDesc_Q( &q, &gradient_d);  // compute q (n+1)

  Dtam::ComputeWeightedDivergenceSobelImage( &q, &gradient_q ); // compute gradient of q (n+1)

  Dtam::gradDesc_D( &d, &a, &gradient_q);  // compute d (n+1)


  Dtam::search_A( &d, &a );  // compute a (n+1)

  Dtam::UpdateParametersReg(false);


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



  camera_vector_cpu_[index_r_]->invdepth_map_gpu_= a;


  double t_e=getTime();
  double delta=t_e-t_s;
  std::cerr << "Regularize took: " << delta << " ms " << std::endl;

}

void Dtam::UpdateState(){

  double t_s=getTime();

  int cols = camera_vector_cpu_[index_r_]->invdepth_map_->image_.cols;
  int rows = camera_vector_cpu_[index_r_]->invdepth_map_->image_.rows;
  float depth1_r=camera_vector_cpu_[index_r_]->min_depth_;
  float depth2_r=camera_vector_cpu_[index_r_]->max_depth_;

  dim3 threadsPerBlock( 32 , 32 , 1);
  dim3 numBlocks( rows/32, cols/32 , 1);
  UpdateState_kernel<<<numBlocks,threadsPerBlock>>>( points_added_, cost_volume_, a, d,  gradient_q,invdepth_r_array_, switch_idx_, switch_depth_, depth1_r, depth2_r);
  printCudaError("State Update State");

  double t_e=getTime();
  double delta=t_e-t_s;
  std::cerr << "State Update: " << delta << " ms " << std::endl;

}

void Dtam::depthSampling(Environment_gpu* environment){
  int rows = environment->resolution_/environment->aspect_;
  int cols = environment->resolution_;
  float depth1_r=environment->min_depth_;
  float depth2_r=environment->max_depth_;
  float* invdepth_r_array_h = new float[NUM_INTERPOLATIONS];

  switch_idx_=40;
  switch_depth_=2;
  for (int i=0; i<switch_idx_; i++){
    float ratio_depth_r = (float)(i)/((float)switch_idx_);
    float depth_r = depth1_r+ratio_depth_r*(switch_depth_-depth1_r);
    invdepth_r_array_h[i]=1.0/depth_r;
    std::cout << "depth: " << depth_r  << ", idx: " << i << std::endl;
  }
  for (int i=switch_idx_; i<NUM_INTERPOLATIONS; i++){
    float ratio_depth_r = (float)(i-switch_idx_)/((float)NUM_INTERPOLATIONS-switch_idx_-1);
    float invdepth_r = (1.0/switch_depth_)+ratio_depth_r*((1.0/depth2_r)-(1.0/switch_depth_));
    float depth_r = 1.0/invdepth_r;
    invdepth_r_array_h[i]=1.0/depth_r;
    std::cout << "depth: " << depth_r  << ", idx: " << i << std::endl;
  }

  cudaError_t err ;

  cudaMalloc(&invdepth_r_array_, sizeof(float)*NUM_INTERPOLATIONS);
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("cudaMalloc (dtam constr) Error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(invdepth_r_array_, invdepth_r_array_h, sizeof(float)*NUM_INTERPOLATIONS, cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("cudaMemcpy (dtam constr) Error: %s\n", cudaGetErrorString(err));


  delete (invdepth_r_array_h);

}

void Dtam::updateDepthMap_gpu(Environment_gpu* environment){

  double t_start;  // time start for computing computation time
  double waitKeyDelay=0;

  bool frame_available=true;
  bool set_reference=true;
  bool init=true;
  int it=0;
  frames_computed_=0;

  depthSampling(environment);

  t_start=getTime();
  while (true){

    // considering 30 fps camera
    float fps=30;
    int current_frame=int((getTime()-t_start-waitKeyDelay)/((1.0/fps)*1000));
    int frames_delta=current_frame-frames_computed_;
    if(frames_delta>=0){
      frame_available=true;
      if(frames_computed_>=environment->camera_vector_cpu_.size()){
        break;
      }
      // load camera (already with pose)
      Dtam::addCamera(environment->camera_vector_cpu_[frames_computed_],environment->camera_vector_gpu_[frames_computed_]);
      frames_computed_+=(frames_delta+1);
      if (frames_delta>0)
        std::cout << frames_delta+1 << " frames has been skipped!" << std::endl;
    }

    if (frame_available){

      if (set_reference){
        Dtam::setReferenceCamera(frames_computed_-1);
        set_reference=false;
      }
      else{
        int index_m=frames_computed_-1;
        // printf("%i\n", index_m);
        Dtam::prepareCameraForDtam(index_m);
        Dtam::UpdateCostVolume(index_m);

        if(index_m==(index_r_+1)){
          Dtam::ComputeCostVolumeMin();

          d = camera_vector_cpu_[index_r_]->invdepth_map_gpu_.clone();
          a = camera_vector_cpu_[index_r_]->invdepth_map_gpu_.clone();
          q.create(d.rows*2,d.cols*2,CV_32FC1);
          gradient_d.create(d.rows*2,d.cols*2,CV_32FC1);
          gradient_q.create(d.rows,d.cols,CV_32FC1);


          init = false;
        }
        else if(index_m>(index_r_+2)){
          UpdateState();
        }

      }


      frame_available=false;
    }
    else if(!init){

      float cr=0.484; float rr=0.465;  //occlusion
      // float cr=0.61; float rr=0.53;  //hightex1
      // float cr=0.61; float rr=0.53;  //hightex2
      // float cr=0.95; float rr=0.87;  //corner dr
      // float cr=0.95; float rr=0.08;  //corner ur
      // float cr=0.5; float rr=0.9;  //hightex cube

      int index_m=frames_computed_-1;
      int col=cr*camera_vector_cpu_[0]->resolution_;
      int row=rr*camera_vector_cpu_[0]->resolution_/camera_vector_cpu_[0]->aspect_;
      // Dtam::StudyCostVolumeMin(index_m, camera_data_for_dtam_, row, col, true);

      if(theta_>theta_end_){
      //   // if(count_>=3){
          Dtam::Regularize();
      //   // }
      }
      double t_s=getTime();
      Dtam::showImgs(640);
      cv::waitKey(0);
      double t_e=getTime();
      double delta=t_e-t_s;
      waitKeyDelay+=delta;

    }


  }

}
