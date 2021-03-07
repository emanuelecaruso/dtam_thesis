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

  Camera_cpu* camera_r = camera_vector_cpu_[index_r];
  Eigen::Vector3f camera_r_p = camera_r->frame_camera_wrt_world_->translation();

  for (int camera_iterator=0; camera_iterator<num_cameras; camera_iterator++){

    if (camera_iterator!=index_r){
      Camera_cpu* camera_m = camera_vector_cpu_[camera_iterator];

      // project camera_r on camera_m
      Eigen::Vector2f cam_r_projected_on_cam_m;
      float cam_r_depth_on_camera_m;
      bool cam_r_in_front = camera_m->projectPoint(camera_r_p, cam_r_projected_on_cam_m, cam_r_depth_on_camera_m);

      Eigen::Isometry3f T = (*(camera_r->frame_world_wrt_camera_))*(*(camera_m->frame_camera_wrt_world_));
      Eigen::Matrix3f r=T.linear();
      Eigen::Vector3f t=T.translation();

      camera_m->T_r=r;
      camera_m->T_t=t;
      camera_m->cam_r_projected_on_cam_m=cam_r_projected_on_cam_m;
      camera_m->cam_r_depth_on_camera_m=cam_r_depth_on_camera_m;
      camera_m->cam_r_in_front=cam_r_in_front;
      //
      Camera_gpu* camera_gpu_h = new Camera_gpu(camera_m->name_, camera_m->lens_, camera_m->aspect_, camera_m->width_, camera_m->resolution_,
         camera_m->max_depth_, camera_m->K_, camera_m->Kinv_, *(camera_m->frame_camera_wrt_world_), *(camera_m->frame_world_wrt_camera_),
          camera_m->depth_map_gpu_, camera_m->image_rgb_gpu_);

      camera_gpu_h->T_r=r;
      camera_gpu_h->T_t=t;
      camera_gpu_h->cam_r_projected_on_cam_m=cam_r_projected_on_cam_m;
      camera_gpu_h->cam_r_depth_on_camera_m=cam_r_depth_on_camera_m;
      camera_gpu_h->cam_r_in_front=cam_r_in_front;

      cudaError_t err ;

      cudaMemcpy(camera_vector_gpu_[camera_iterator], camera_gpu_h, sizeof(Camera_gpu), cudaMemcpyHostToDevice);
      err = cudaGetLastError();
      if (err != cudaSuccess)
          printf("cudaMemcpy (dtam constr) %s%s",camera_m->name_," Error: %s\n", cudaGetErrorString(err));
      //
      // camera_vector_gpu_[index_r]=camera_gpu_d;
      //
      delete camera_gpu_h;
    }

  }
  return true;

}
// bool Dtam::get1stDepthWithUV(Camera* camera_r, Camera* camera_m, Eigen::Vector2f& uv_r, Eigen::Vector2f& uv_m, float& depth){
//
//   Eigen::Isometry3f T = (*(camera_m->frame_world_wrt_camera_))*(*(camera_r->frame_camera_wrt_world_));
//   auto r=T.linear();
//   auto t=T.translation();
//   float f = camera_r->lens_;
//   float w=camera_m->width_;
//   float h=camera_m->width_/camera_m->aspect_;
//   depth = (2*f*(f+t(2)))/(2*f*r(2,2)-2*r(2,0)*uv_r.x()+r(2,0)*w-r(2,1)*(h-2*uv_r.y()));
//   uv_m.x() = t(0)+(w/2)-depth*r(0,2)+((depth*r(0,0)*(2*uv_r.x()-w))/(2*f))+((depth*r(0,1)*(h-2*uv_r.y()))/(2*f));
//   uv_m.y() = (h/2)-t(1)+depth*r(1,2)-((depth*r(1,0)*(2*uv_r.x()-w))/(2*f))-((depth*r(1,1)*(h-2*uv_r.y()))/(2*f));
//   return true;
// }

__global__ void ComputeCostVolume_kernel(Camera_gpu* camera_r, Camera_gpu* camera_m, int num_interpolations,
            cv::cuda::PtrStepSz<int> cost_matrix,cv::cuda::PtrStepSz<uchar> n_valid_proj_matrix){

  //cv::cuda::PtrStepSz<uchar3> dOutput

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  int i = blockIdx.z * blockDim.z + threadIdx.z;

  // initializations
  Eigen::Vector2f uv1, uv2;
  float depth1_m, depth2_m;
  bool resized1, resized2;
  float depth1_r=camera_r->lens_;
  float depth2_r=camera_r->max_depth_;
  Eigen::Vector2f uv_r;
  uchar3 clr_r = camera_r->image_rgb_(row,col);

  __shared__ Eigen::Vector2f uv1_fixed, uv2_fixed;
  __shared__ float depth1_m_fixed, depth2_m_fixed;

  bool stop = false;

  if (i==0){
    Eigen::Vector2i pixel_coords_r(col,row);

    // query point
    Eigen::Vector3f query_p;

    camera_r->pixelCoords2uv(pixel_coords_r, uv_r);
    camera_r->pointAtDepth(uv_r, depth2_r, query_p);

    bool invalid_pxl=false;

    auto cam_r_projected_on_cam_m = camera_m->cam_r_projected_on_cam_m;
    auto cam_r_depth_on_camera_m = camera_m->cam_r_depth_on_camera_m;
    auto cam_r_in_front = camera_m->cam_r_in_front;

    // project query point
    Eigen::Vector2f query_p_projected_on_cam_m;
    float query_depth_on_camera_m;
    bool query_in_front = camera_m->projectPoint(query_p, query_p_projected_on_cam_m, query_depth_on_camera_m);


    // if both camera r and query point are on back of camera m return false
    if (!query_in_front && !cam_r_in_front)
      stop=true;
    // if query point is in front of camera m whereas camera r is on the back
    else if (query_in_front && !cam_r_in_front){
      // std::cout << "query in front" << std::endl;
      // uv2=query_p_projected_on_cam_m;
      // depth2=query_depth_on_camera_m;
      // Dtam::get1stDepthWithUV(camera_r, camera_m, uv_r, uv1, depth1);
      // depth1=camera_r->lens_;
      // bool flag = camera_m->resizeLine(uv1 , uv2, depth1, depth2);
      // if (!flag)
      //   return false;
      uv1_fixed=cam_r_projected_on_cam_m;
      uv2_fixed=query_p_projected_on_cam_m;
      depth1_m_fixed=cam_r_depth_on_camera_m;
      depth2_m_fixed=query_depth_on_camera_m;

    }
    // if camera r is in front of camera m whereas query point is on the back
    else if (!query_in_front && cam_r_in_front){
      // TODO
      stop=true;
    }
    // if both camera r and query point are in front of camera m
    else {

      // std::cout << "both in front" << std::endl;
      // uv1=cam_r_projected_on_cam_m;
      // uv2=query_p_projected_on_cam_m;
      uv1_fixed=cam_r_projected_on_cam_m;
      uv2_fixed=query_p_projected_on_cam_m;
      // depth1_m=cam_r_depth_on_camera_m;
      // depth2_m=query_depth_on_camera_m;
      depth1_m_fixed=cam_r_depth_on_camera_m;
      depth2_m_fixed=query_depth_on_camera_m;
      // invalid_pxl = camera_m->resizeLine(uv1 , uv2, depth1_m, depth2_m, resized1, resized2);

      // if (!invalid_pxl)
      //   continue;
    }

  }

  __syncthreads();

  Eigen::Vector2i pixel_current;

  if(!stop){

    Eigen::Matrix3f r=camera_m->T_r;
    Eigen::Vector3f t=camera_m->T_t;
    float f = camera_m->lens_;
    float w=camera_m->width_;
    float h=camera_m->width_/camera_m->aspect_;

    Eigen::Vector2f uv_current;

    float ratio_depth_r = (float)i/((float)num_interpolations-1);
    float depth_r = depth1_r+ratio_depth_r*(depth2_r-depth1_r);

    float depth_m = depth_r*r(2,2)-t(2)-((depth_r*r(2,0)*(2*uv_r.x()-w))/(2*f))-((depth_r*r(2,1)*(-2*uv_r.y()+h))/(2*f));

    float ratio_invdepth_m = ((1.0/depth_m)-(1.0/depth1_m_fixed))/((1.0/depth2_m_fixed)-(1.0/depth1_m_fixed));

    // std::cout << (1.0/depth1_m_fixed) << std::endl;



    uv_current.x()=uv1_fixed.x()+ratio_invdepth_m*(uv2_fixed.x()-uv1_fixed.x()) ;
    uv_current.y()=uv1_fixed.y()+ratio_invdepth_m*(uv2_fixed.y()-uv1_fixed.y()) ;


    camera_m->uv2pixelCoords( uv_current, pixel_current);

    if(pixel_current.x()<0 || pixel_current.y()<0 || pixel_current.x()>(camera_m->resolution_) || pixel_current.x()>((float)camera_m->resolution_/(float)camera_m->aspect_) )
      stop=true;
  }

  if (!stop){
    uchar3 clr_current = camera_m->image_rgb_(pixel_current.y(),pixel_current.x());

    int cost_current=((clr_r.x-clr_current.x)*(clr_r.x-clr_current.x)+(clr_r.y-clr_current.y)*(clr_r.y-clr_current.y)+(clr_r.z-clr_current.z)*(clr_r.z-clr_current.z));
    // int cost_current=(abs(clr_r.x-clr_current.x)+abs(clr_r.y-clr_current.y)+abs(clr_r.z-clr_current.z));


    int col_ = camera_m->resolution_*i+col;

    // cost_matrix(row,col_) = 1;
    if (cost_matrix(row,col_)==999999){
      cost_matrix(row,col_) = cost_current;
    }
    else
      cost_matrix(row,col_) = (cost_matrix(row,col_)*n_valid_proj_matrix(row,col_)+cost_current)/(n_valid_proj_matrix(row,col_)+1);


    n_valid_proj_matrix(row,col_)+=1;

    // // cv::Vec3b clr_current;
    // // bool flag = camera_m->image_rgb_->evalPixel(pixel_current,clr_current);
    // // bool flag = camera_m->image_rgb_->evalPixel(pixel_current,clr_current);
    //
    // //
    // // num_valid_projections++;
    // //
    // // int cost_current = mseBetween2Colors(clr_r, clr_current);
  }


  // // (*d_depth_map).at<float>(0,0) = 0.0;
  // // d_depth_map[0] = 0.0;
  // // printf("\n");
  // // printf(d_depth_map[0]);
  // // printf("\n");
  // // d_depth_map[0] = static_cast<unsigned char>(0.0);
  // // dOutput(1, 1)=0.0;
  // dOutput(row, col).x = 0;
  // dOutput(row, col).y = 0;
  // dOutput(row, col).z = 255;
  // // a.x=1;
  // // printf("\n");
  // // printf(a.val[0]);
  // // printf("\n");
  //
  // // struct cameraData camera_data;
  // // camera_data = d_cameraData_vector[0];  //index into array

}

__global__ void ComputeDepthMap(Camera_gpu* camera_r, int num_interpolations,
        cv::cuda::PtrStepSz<int> cost_matrix, cv::cuda::PtrStepSz<uchar> n_valid_proj_matrix){

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  int i = blockIdx.z * blockDim.z + threadIdx.z;


  extern __shared__ int cost_array[];  //TODO change to num num_interpolations

  cost_array[i]=cost_matrix(row,camera_r->resolution_*i+col);

  __syncthreads();

  // TODO this may be inefficient
  if (i==0){
    int min_value=999999;
    int min_index=num_interpolations;
    for (int j=0; j<num_interpolations; j++){
      if (cost_array[j]<min_value){
        min_value=cost_array[j];
        min_index=j;
      }
    }

    float ratio = (float)min_index/(float)num_interpolations;
    camera_r->depth_map_(row,col)=(camera_r->lens_+(ratio*(camera_r->max_depth_-camera_r->lens_)))/(camera_r->max_depth_);
    // camera_r->depth_map_(row,col)=0;
  }

}

void Dtam::getDepthMap(int num_interpolations, bool check){

  cudaError_t err ;

  // // reference camera
  Camera_cpu* camera_r_cpu = camera_vector_cpu_[index_r_];
  Camera_gpu* camera_r_gpu = camera_vector_gpu_[index_r_];
  int cols = camera_r_cpu->depth_map_->image_.cols;
  int rows = camera_r_cpu->depth_map_->image_.rows;

  cost_matrix_.create(rows,cols*num_interpolations,CV_32SC1);
  cost_matrix_.setTo(cv::Scalar::all(999999));
  n_valid_proj_matrix_.create(rows,cols*num_interpolations,CV_8UC1);
  n_valid_proj_matrix_.setTo(cv::Scalar::all(0));



  // int num_cameras = camera_vector_gpu_.size();
  // Camera_gpu** camera_gpu_array_h = &camera_vector_gpu_[0];
  // Camera_gpu** camera_gpu_array_d;
  // cudaMalloc(&camera_gpu_array_d, sizeof(Camera_gpu*)*num_cameras);
  // err = cudaGetLastError();
  // if (err != cudaSuccess)
  //     printf("cudaMalloc camgpu in getDepthMap Error: %s\n", cudaGetErrorString(err));
  //   camera_r->depth_map_(row,col)=0.0;
  //
  // cudaMemcpy(camera_gpu_array_d, camera_gpu_array_h, sizeof(Camera_gpu*)*num_cameras, cudaMemcpyHostToDevice);
  // err = cudaGetLastError();
  // if (err != cudaSuccess)
  //     printf("cudaMemcpy camgpu in getDepthMap Error: %s\n", cudaGetErrorString(err));



  // Kernel invocation for computing cost volume
  dim3 threadsPerBlock( 1 , 1 , num_interpolations);
  dim3 numBlocks( rows, cols , 1);

  for (int i=0; i<camera_vector_gpu_.size(); i++){
    if (i!=index_r_)
      ComputeCostVolume_kernel<<<numBlocks,threadsPerBlock>>>(camera_vector_gpu_[index_r_], camera_vector_gpu_[i], num_interpolations, cost_matrix_ , n_valid_proj_matrix_);
      err = cudaGetLastError();
      if (err != cudaSuccess)
          printf("Kernel computing cost volume Error: %s\n", cudaGetErrorString(err));

  }

  cudaDeviceSynchronize();

  // Kernel invocation for computing depth map
  ComputeDepthMap<<<numBlocks,threadsPerBlock,num_interpolations*sizeof(int)>>>(camera_vector_gpu_[index_r_], num_interpolations, cost_matrix_ , n_valid_proj_matrix_);

  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("Kernel computing depth map Error: %s\n", cudaGetErrorString(err));

  camera_r_cpu->depth_map_gpu_.download(camera_r_cpu->depth_map_->image_);


  //
  //
  // Eigen::Vector3f camera_r_p = camera_r->frame_camera_wrt_world_->translation();
  // float depth1_r=camera_r->lens_;
  // float depth2_r=camera_r->max_depth_;
  //

  //
  //
  // float f = camera_r->lens_;
  // float w=camera_r->width_;
  // float h=camera_r->width_/camera_r->aspect_;



  // cudaDeviceSynchronize();




  // for (int row = 0; row<rows; row++){
  //   for (int col = 0; col<cols; col++){
  //
  //     if (check){row= rows*0.36; col=cols*0.45;}
  //     // if (check){row= rows/3; col=1;}
  //     // if (row==1 && col==0)
  //     //   check=true;
  //
  //     Eigen::Vector2i pixel_coords_r(col,row);
  //     cv::Vec3b clr_r;
  //     camera_r->image_rgb_->evalPixel(pixel_coords_r,clr_r);
  //
  //
  //     // query point
  //     Eigen::Vector3f query_p;
  //     Eigen::Vector2f uv_r;
  //
  //     camera_r->pixelCoords2uv(pixel_coords_r, uv_r);
  //     camera_r->pointAtDepth(uv_r, depth2_r, query_p);
  //
  //     bool invalid_pxl=false;
  //
  //     for (int camera_iterator=1; camera_iterator<camera_vector_cpu.size(); camera_iterator++){
  //       Camera* camera_m = camera_vector_cpu[camera_iterator];
  //
  //       auto cam_r_projected_on_cam_m = cameraData_vector_host[camera_iterator-1]->cam_r_projected_on_cam_m;
  //       auto cam_r_depth_on_camera_m = cameraData_vector_host[camera_iterator-1]->cam_r_depth_on_camera_m;
  //       auto cam_r_in_front = cameraData_vector_host[camera_iterator-1]->cam_r_in_front;
  //
  //       // project query point
  //       Eigen::Vector2f query_p_projected_on_cam_m;
  //       float query_depth_on_camera_m;
  //       bool query_in_front = camera_m->projectPoint(query_p, query_p_projected_on_cam_m, query_depth_on_camera_m);
  //
  //       // initializations
  //       Eigen::Vector2f uv1, uv2, uv1_fixed, uv2_fixed;
  //       float depth1_m, depth2_m, depth1_m_fixed, depth2_m_fixed;
  //       bool resized1, resized2;
  //
  //       if (check){ cv::Vec3b clr = cv::Vec3b(0,0,255);
  //         camera_r->image_rgb_->setPixel(pixel_coords_r,clr); }
  //
  //       // if both camera r and query point are on back of camera m return false
  //       if (!query_in_front && !cam_r_in_front)
  //         continue;
  //       // if query point is in front of camera m whereas camera r is on the back
  //       else if (query_in_front && !cam_r_in_front){
  //         // std::cout << "query in front" << std::endl;
  //         // uv2=query_p_projected_on_cam_m;
  //         // depth2=query_depth_on_camera_m;
  //         // Dtam::get1stDepthWithUV(camera_r, camera_m, uv_r, uv1, depth1);
  //         // depth1=camera_r->lens_;
  //         // bool flag = camera_m->resizeLine(uv1 , uv2, depth1, depth2);
  //         // if (!flag)
  //         //   return false;
  //         continue;
  //
  //       }
  //       // if camera r is in front of camera m whereas query point is on the back
  //       else if (!query_in_front && cam_r_in_front){
  //         // TODO
  //         continue;
  //       }
  //       // if both camera r and query point are in front of camera m
  //       else {
  //
  //         // std::cout << "both in front" << std::endl;
  //         // uv1=cam_r_projected_on_cam_m;
  //         uv2=query_p_projected_on_cam_m;
  //         // uv1_fixed=cam_r_projected_on_cam_m;
  //         uv2_fixed=query_p_projected_on_cam_m;
  //         // depth1_m=cam_r_depth_on_camera_m;
  //         depth2_m=query_depth_on_camera_m;
  //         // depth1_m_fixed=cam_r_depth_on_camera_m;
  //         depth2_m_fixed=query_depth_on_camera_m;
  //         // invalid_pxl = camera_m->resizeLine(uv1 , uv2, depth1_m, depth2_m, resized1, resized2);
  //
  //         // cameraData_vector_host[camera_iterator-1]->uv1=uv1;
  //         cameraData_vector_host[camera_iterator-1]->uv2=uv2;
  //         // cameraData_vector_host[camera_iterator-1]->uv1_fixed=uv1_fixed;
  //         cameraData_vector_host[camera_iterator-1]->uv2_fixed=uv2_fixed;
  //         cameraData_vector_host[camera_iterator-1]->depth1_m=depth1_m;
  //         cameraData_vector_host[camera_iterator-1]->depth2_m=depth2_m;
  //         // cameraData_vector_host[camera_iterator-1]->depth1_m_fixed=depth1_m_fixed;
  //         cameraData_vector_host[camera_iterator-1]->depth2_m_fixed=depth2_m_fixed;
  //
  //         if (!invalid_pxl)
  //           continue;
  //
  //
  //       }
  //
  //     }
  //
  //
  //
  //     int cost_min = 999999;
  //     float depth_min = -1;
  //     // int iterator = 0;
  //
  //
  //     for (int i=0; i<num_interpolations; i++){
  //     // for (int i=0; i<5; i++){
  //
  //       float ratio_depth_r = (float)i/((float)num_interpolations-1);
  //       float depth_r = depth1_r+ratio_depth_r*(depth2_r-depth1_r);
  //
  //       int cost_i = -1;
  //       int num_valid_projections = 0;
  //
  //
  //       for (int camera_iterator=1; camera_iterator<camera_vector_cpu.size(); camera_iterator++){
  //         Camera* camera_m = camera_vector_cpu[camera_iterator];
  //
  //
  //         auto uv1=cameraData_vector_host[camera_iterator-1]->uv1;
  //         auto uv2=cameraData_vector_host[camera_iterator-1]->uv2;
  //         auto uv1_fixed=cameraData_vector_host[camera_iterator-1]->uv1_fixed;
  //         auto uv2_fixed=cameraData_vector_host[camera_iterator-1]->uv2_fixed;
  //         auto depth1_m=cameraData_vector_host[camera_iterator-1]->depth1_m;
  //         auto depth2_m=cameraData_vector_host[camera_iterator-1]->depth2_m;
  //         auto depth1_m_fixed=cameraData_vector_host[camera_iterator-1]->depth1_m_fixed;
  //         auto depth2_m_fixed=cameraData_vector_host[camera_iterator-1]->depth2_m_fixed;
  //         auto r=cameraData_vector_host[camera_iterator-1]->r;
  //         auto t=cameraData_vector_host[camera_iterator-1]->t;
  //
  //
  //         Eigen::Vector2f uv_current;
  //
  //
  //         float depth_m = depth_r*r(2,2)-t(2)-((depth_r*r(2,0)*(2*uv_r.x()-w))/(2*f))-((depth_r*r(2,1)*(-2*uv_r.y()+h))/(2*f));
  //
  //
  //         // if (depth_m< depth1_m || depth_m> depth2_m){
  //         //   // if (check)
  //         //     std::cout << "row " << row << " col " << col << " i "<< i << " cost_i " << cost_i << std::endl;
  //         //   continue;}
  //
  //         // float invdepth_m = 1.0/depth_m;
  //         // float invdepth_delta = (1.0/depth2_m_fixed)-(1.0/depth1_m_fixed);
  //         float ratio_invdepth_m = ((1.0/depth_m)-(1.0/depth1_m_fixed))/((1.0/depth2_m_fixed)-(1.0/depth1_m_fixed));
  //
  //         // std::cout << (1.0/depth1_m_fixed) << std::endl;
  //
  //
  //
  //         uv_current.x()=uv1_fixed.x()+ratio_invdepth_m*(uv2_fixed.x()-uv1_fixed.x()) ;
  //         uv_current.y()=uv1_fixed.y()+ratio_invdepth_m*(uv2_fixed.y()-uv1_fixed.y()) ;
  //
  //         Eigen::Vector2i pixel_current;
  //         camera_m->uv2pixelCoords( uv_current, pixel_current);
  //         cv::Vec3b clr_current;
  //         bool flag = camera_m->image_rgb_->evalPixel(pixel_current,clr_current);
  //         if (!flag){
  //           // if (check)
  //             // std::cout << "row " << row << " col " << col << " i "<< i << " cost_i " << cost_i << std::endl;
  //           continue;}
  //
  //         num_valid_projections++;
  //
  //         int cost_current = mseBetween2Colors(clr_r, clr_current);
  //
  //
  //         if (cost_i<0)
  //           cost_i=0;
  //
  //         cost_i+=cost_current;
  //
  //         if (check){
  //           // std::cout << uv_current.x() << " " << uv_current.y() << std::endl;
  //           cv::Vec3b clr = cv::Vec3b(255,0,255);
  //           bool fl = camera_m->image_rgb_->setPixel(pixel_current,clr);
  //           // std::cout << << std::endl;
  //           // if (fl)
  //           // std::cout << ratio_invdepth_m << std::endl;
  //         }
  //
  //       }
  //
  //
  //       if (cost_i>0 && cost_i<cost_min){
  //         depth_min=depth_r;
  //         cost_min=cost_i/num_valid_projections;
  //         // iterator=i;
  //       }
  //     }
  //
  //
  //
  //     // float depth_min = (depth1_r+(float)iterator/((float)num_interpolations-1)*(depth2_r-depth1_r));
  //     // std::cout << depth_min << std::endl;
  //
  //     if (depth_min<0){
  //       // std::cout << "row-col " << row << " " << col << std::endl;
  //
  //       camera_r->depth_map_->setPixel(pixel_coords_r,1.0);
  //     }
  //     else{
  //       // std::cout << "row-col " << row << " " << col << ", depth "<< depth_min << ", cost " << cost_min << std::endl;
  //       // std::cout << pixel_coords_r << std::endl;
  //       float depth_value = depth_min/camera_r->max_depth_;
  //       camera_r->depth_map_->setPixel(pixel_coords_r,depth_value);
  //     }
  //
  //
  //
  //
  //     // for (int i=0; i< num_interpolations; i++){
  //     //   if (costs[i]>=0){
  //     //     if (costs[i]<cost_min){
  //     //       cost_min=costs[i];
  //     //       depth_min = (depth1_r+(float)i/((float)num_interpolations-1)*(depth2_r-depth1_r));
  //     //     }
  //     //   }
  //     // }
  //     // float depth_value = depth_min/camera_vector_cpu[0]->max_depth_;
  //     // if (depth_min==0)
  //     //   camera_vector_cpu[0]->depth_map_->setPixel(pixel_coords_r,1.0);
  //     // else
  //     //   camera_vector_cpu[0]->depth_map_->setPixel(pixel_coords_r,depth_value);
  //
  //
  //     if (check)  {break;}
  //   }
  //   if (check)  {break;}
  // }


}
