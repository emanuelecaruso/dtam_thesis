#include "dtam_cuda.cuh"
// #include "cuda_handler.cuh"
#include <math.h>
#include "utils.h"
#include <stdlib.h>


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

__global__ void CostVolumeMin_kernel(cv::cuda::PtrStepSz<uchar3> dOutput, cameraData* d_cameraData_vector, int n_cameras){
  // int row = blockIdx.x * blockDim.x + threadIdx.x;
  // int col = blockIdx.y * blockDim.y + threadIdx.y;
  //
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

void Dtam::CostVolumeMin(CameraVector_gpu camera_vector_gpu, cameraData* d_cameraData_vector, int n_cameras){


  // // int* h_msg = (int*)malloc(sizeof(int));
  // // *h_msg = -1;
  // // // int msg = *h_msg;
  // //
  // cv::Mat_<cv::Vec3b> depth_map = camera_vector_gpu[0]->image_rgb_->image_;
  // cv::cuda::GpuMat depth_map_gpu;
  // depth_map_gpu.upload(depth_map);
  //
  //
  // // auto size = sizeof(depth_map);
  // // float* d_depth_map;
  // // unsigned char* h_depth_map = depth_map.ptr();
  //
  // // cudaMalloc(&d_depth_map, size);
  // // cudaMemcpy(d_depth_map, h_depth_map, size, cudaMemcpyHostToDevice);
  // //

  // Kernel invocation
  const int N = 1;
  dim3 threadsPerBlock( camera_vector_gpu[0]->width_/32, 32, 100);
  dim3 numBlocks( (camera_vector_gpu[0]->width_/camera_vector_gpu[0]->aspect_) / 32, 100);

  // for (int i=0; i<camera_vector_gpu.size(); i++){
  //   CostVolumeMin_kernel<<<numBlocks,threadsPerBlock>>>(camera_vector_gpu[i], d_cameraData_vector, n_cameras);
  // }
  // CostVolumeMin_kernel<<<1,1>>>(depth_map_gpu, d_cameraData_vector, n_cameras);
  //
  // depth_map_gpu.download(depth_map);
  //
  // depth_map_gpu.release();
  // //
  // // cudaMemcpy(h_depth_map, d_depth_map, grayBytes, cudaMemcpyDeviceToHost);
  // // cudaFree(d_depth_map);
  //
  // //
  // // std::cout << *h_msg << std::endl;

}

void Dtam::getDepthMap(int num_interpolations, CameraVector_cpu& camera_vector_cpu, CameraVector_gpu& camera_vector_gpu, bool check){


  int cameraData_size = camera_vector_cpu.size()-1;

  cameraData cameraData_vector[cameraData_size];
  cameraData* d_cameraData_vector;
  cudaMalloc((void**)&d_cameraData_vector, sizeof(struct cameraData) * cameraData_size);


  // reference camera
  Camera_cpu* camera_r = camera_vector_cpu[0];


  Eigen::Vector3f camera_r_p = camera_r->frame_camera_wrt_world_->translation();
  float depth1_r=camera_r->lens_;
  float depth2_r=camera_r->max_depth_;

  int cols = camera_r->depth_map_->image_.cols;
  int rows = camera_r->depth_map_->image_.rows;


  float f = camera_r->lens_;
  float w=camera_r->width_;
  float h=camera_r->width_/camera_r->aspect_;


  for (int camera_iterator=0; camera_iterator<cameraData_size; camera_iterator++){
    cameraData camera_data;

    // project camera_r on camera_m
    Eigen::Vector2f cam_r_projected_on_cam_m;
    float cam_r_depth_on_camera_m;
    bool cam_r_in_front = camera_vector_cpu[camera_iterator+1]->projectPoint(camera_r_p, cam_r_projected_on_cam_m, cam_r_depth_on_camera_m);

    Eigen::Isometry3f T = (*camera_r->frame_world_wrt_camera_)*(*(camera_vector_cpu[camera_iterator+1]->frame_camera_wrt_world_));
    Eigen::Matrix3f r=T.linear();
    Eigen::Vector3f t=T.translation();
    camera_data.r=r;
    camera_data.t=t;
    camera_data.cam_r_projected_on_cam_m=cam_r_projected_on_cam_m;
    camera_data.cam_r_depth_on_camera_m=cam_r_depth_on_camera_m;
    camera_data.cam_r_in_front=cam_r_in_front;

    camera_data.uv1=cam_r_projected_on_cam_m;
    camera_data.uv1_fixed=cam_r_projected_on_cam_m;
    camera_data.depth1_m_fixed=cam_r_depth_on_camera_m;

    cameraData_vector[camera_iterator]=camera_data;

    // cameraData* d_cameraData_vector_device;
    // const size_t sz = size_t(cameraData_size) * sizeof(cameraData);
    // cudaMalloc((void**)&d_cameraData_vector_device, sz);
    // cudaMemcpy(d_cameraData_vector_device, &cameraData_vector_device[0], sz, cudaMemcpyHostToDevice);
  }

  // cudaDeviceSynchronize();
  cudaMemcpy(d_cameraData_vector, &cameraData_vector, sizeof(struct cameraData)* cameraData_size, cudaMemcpyHostToDevice);

  CostVolumeMin(camera_vector_gpu,d_cameraData_vector, cameraData_size);

  cudaFree(d_cameraData_vector);


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
