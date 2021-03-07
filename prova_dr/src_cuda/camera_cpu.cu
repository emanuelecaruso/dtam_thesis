#include "camera_cpu.cuh"
#include <thread>
#include <vector>
#include <mutex>

using namespace std;
using namespace pr;



void Camera_cpu::clearImgs(){
  depth_map_->image_=1.0;
  image_rgb_->image_=cv::Vec3b(255,255,255);
}


void Camera_cpu::gpuFree(){
  image_rgb_gpu_.release();
  depth_map_gpu_.release();
}


Camera_gpu* Camera_cpu::getCamera_gpu(){

  image_rgb_gpu_.upload(image_rgb_->image_);
  depth_map_gpu_.upload(depth_map_->image_);

  Camera_gpu* camera_gpu_h = new Camera_gpu(name_, lens_, aspect_, width_, resolution_,
     max_depth_, K_, Kinv_, *frame_camera_wrt_world_, *frame_world_wrt_camera_,
      depth_map_gpu_, image_rgb_gpu_);

  cudaError_t err ;

  Camera_gpu* camera_gpu_d;
  cudaMalloc((void**)&camera_gpu_d, sizeof(Camera_gpu));
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("cudaMalloc %s%s",name_," Error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(camera_gpu_d, camera_gpu_h, sizeof(Camera_gpu), cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("cudaMemcpy %s%s",name_," Error: %s\n", cudaGetErrorString(err));

  delete camera_gpu_h;
  
  return camera_gpu_d;
}

void Camera_cpu::printMembers(){

  std::cout << "name: " << name_ << std::endl;
  std::cout << "lens: " << lens_ << std::endl;
  std::cout << "aspect: " << aspect_ << std::endl;
  std::cout << "width: " << width_ << std::endl;
  std::cout << "resolution: " << resolution_ << std::endl;
  std::cout << "max_depth: " << max_depth_ << std::endl;
  std::cout << "K: " << K_ << std::endl;
  std::cout << "Kinv: " << Kinv_ << std::endl;
  std::cout << "frame_world_wrt_camera LINEAR:\n" << (*frame_world_wrt_camera_).linear() << std::endl;
  std::cout << "frame_world_wrt_camera TRANSL:\n" << (*frame_world_wrt_camera_).translation() << std::endl;
  std::cout << "frame_camera_wrt_world LINEAR:\n" << (*frame_camera_wrt_world_).linear() << std::endl;
  std::cout << "frame_camera_wrt_world TRANSL:\n" << (*frame_camera_wrt_world_).translation() << std::endl;
  std::cout << "\n" << std::endl;

}


// void Camera_cpu::showWorldFrame(Eigen::Vector3f origin, float delta, int length){
//   Camera_cpu::clearImgs();
//   std::vector<Cp> cps_world_frame;
//   for (int i=0; i<length; i++)
//   {
//     Eigen::Vector3f x(origin[0]+delta*i,origin[1],origin[2]);
//     cv::Vec3b color1(0,0,255);
//     struct Cp cp1 = {x, color1};
//     cps_world_frame.push_back(cp1);
//
//     Eigen::Vector3f y(origin[0],origin[1]+delta*i,origin[2]);
//     cv::Vec3b color2(0,255,0);
//     struct Cp cp2 = {y, color2};
//     cps_world_frame.push_back(cp2);
//
//     if (i>0)
//     {
//       Eigen::Vector3f z(origin[0],origin[1],origin[2]+delta*i);
//       cv::Vec3b color3(255,0,0);
//       struct Cp cp3 = {z, color3};
//       cps_world_frame.push_back(cp3);
//     }
//   }
//
//
//   int cols=resolution_;
//   int rows=resolution_/aspect_;
//
//   for (Cp cp : cps_world_frame){
//     Eigen::Vector3f p_cam = *frame_world_wrt_camera_*cp.point;
//
//     if (p_cam.z()>-lens_)
//       continue;
//
//     Eigen::Vector3f p_proj = K_*p_cam;
//
//     Eigen::Vector2f uv = p_proj.head<2>()*(1./p_proj.z());
//
//     if(uv.x()<0 || uv.x()>width_)
//       continue;
//     if(uv.y()<0 || uv.y()>(width_/aspect_))
//       continue;
//
//     Eigen::Vector2i pixel_coords;
//     Camera_cpu::uv2pixelCoords( uv, pixel_coords);
//
//     if (cp.color[0]>255 || cp.color[1]>255 || cp.color[2]>255)
//       continue;
//
//     int r=pixel_coords.y();
//     int c=pixel_coords.x();
//     if(r<0||r>=rows)
//       continue;
//     if(c<0||c>=cols)
//       continue;
//
//     cv::circle(image_rgb_->image_, cv::Point(c,r), 3, cp.color);
//   }
//
// }

void Camera_cpu::pixelCoords2uv(Eigen::Vector2i& pixel_coords, Eigen::Vector2f& uv){
  float pixel_width = width_/resolution_;

  uv.x()=((float)pixel_coords.x()/(resolution_))*width_+(pixel_width/2);
  uv.y()=(((float)pixel_coords.y())/(float)((resolution_)/aspect_))*(float)(width_/aspect_)+(pixel_width/2);
}

void Camera_cpu::uv2pixelCoords( Eigen::Vector2f& uv, Eigen::Vector2i& pixel_coords){

  pixel_coords.x()=(int)((uv.x()/width_)*resolution_);
  pixel_coords.y()=(int)((uv.y()/(width_/aspect_))*(resolution_/aspect_));
}

void Camera_cpu::pointAtDepth(Eigen::Vector2f& uv, float depth, Eigen::Vector3f& p){

  Eigen::Vector3f p_proj;
  Eigen::Vector2f product = uv * depth;
  p_proj.x() = product.x();
  p_proj.y() = product.y();
  p_proj.z() = depth;
  Eigen::Vector3f p_cam = Kinv_*p_proj;
  p = *frame_camera_wrt_world_*p_cam;

}

bool Camera_cpu::projectPoint(Eigen::Vector3f& p, Eigen::Vector2f& uv, float& p_cam_z ){


  Eigen::Vector3f p_cam = *frame_world_wrt_camera_*p;

  // return wether the projected point is in front or behind the camera
  p_cam_z=-p_cam.z();

  Eigen::Vector3f p_proj = K_*p_cam;

  uv = p_proj.head<2>()*(1./p_proj.z());

  if (p_cam_z<lens_)
    return false;

  return true;

}

//
// bool Camera_cpu::resizeLine(Eigen::Vector2f& uv1 ,Eigen::Vector2f& uv2, float& depth1, float& depth2, bool& resized1, bool& resized2){
//
//   float pixel_width= width_/resolution_;
//   float height = width_/aspect_;
//
//   float delta_x = uv2.x()-uv1.x();
//   float delta_y = uv2.y()-uv1.y();
//   float steepness=delta_y/delta_x;
//   float invdepth_delta=(1.0/depth2)-(1.0/depth1);
//
//   bool top1 = false; bool bottom1 = false; bool left1 = false; bool right1 = false;
//   bool top2 = false; bool bottom2 = false; bool left2 = false; bool right2 = false;
//   resized1 = false;
//   resized2 = false;
//
//   if ( uv1.x() < pixel_width/2 )
//     left1=true;
//   else if (uv1.x() > width_-(pixel_width/2))
//     right1=true;
//   if ( uv1.y() < pixel_width/2 )
//     top1=true;
//   else if (uv1.y() > height-(pixel_width/2))
//     bottom1=true;
//   if ( uv2.x() < pixel_width/2 )
//     left2=true;
//   else if (uv2.x() > width_-(pixel_width/2))
//     right2=true;
//   if ( uv2.y() < pixel_width/2 )
//     top2=true;
//   else if (uv2.y() > height-(pixel_width/2))
//     bottom2=true;
//
//   if ( (left1&&left2) || (right1&&right2) || (top1&&top2) || (bottom1&&bottom2) )
//     return false;
//
//   if (left1)
//   {
//     float deltax1 = -uv1.x()+(pixel_width/2);
//
//     float v=uv1.y()+steepness*deltax1;
//     if (v>=0 && v<=height)
//     {
//       float ratio_x = deltax1/delta_x;
//       float invdepth = ratio_x*invdepth_delta;
//       depth1 = 1.0/((1.0/depth1)+invdepth);
//
//       uv1.x()=(pixel_width/2);
//       uv1.y()=v;
//       resized1=true;
//
//     }
//
//   }
//   if (top1 && !resized1)
//   {
//     float deltay1=-uv1.y()+(pixel_width/2);
//
//     float u=uv1.x()+(1.0/steepness)*deltay1;
//     if (u>=0 && u<=width_)
//     {
//       float ratio_y = deltay1/delta_y;
//       float invdepth = ratio_y*invdepth_delta;
//       depth1 = 1.0/((1.0/depth1)+invdepth);
//       uv1.x()=u;
//       uv1.y()=(pixel_width/2);
//       resized1=true;
//       // if (depth1>depth2)
//       //   std::cout << "top" << std::endl;
//     }
//   }
//   if (right1 && !resized1)
//   {
//     float deltax1 = width_-(pixel_width/2)-uv1.x();
//
//     float v=uv1.y()+steepness*deltax1;
//     if (v>=0 && v<=height)
//     {
//       float ratio_x = deltax1/delta_x;
//       float invdepth = ratio_x*invdepth_delta;
//       depth1 = 1.0/((1.0/depth1)+invdepth);
//       uv1.x()=width_-(pixel_width/2);
//       uv1.y()=v;
//       resized1=true;
//       // if (depth1>depth2)
//       //   std::cout << "right" << std::endl;
//     }
//   }
//   if (bottom1 && !resized1)
//   {
//     float deltay1=height-(pixel_width/2)-uv1.y();
//
//     float u=uv1.x()+(1.0/steepness)*deltay1;
//     if (u>=0 && u<=width_)
//     {
//       float ratio_y = deltay1/delta_y;
//       float invdepth = ratio_y*invdepth_delta;
//       depth1 = 1.0/((1.0/depth1)+invdepth);
//       uv1.x()=u;
//       uv1.y()=height-(pixel_width/2);
//       resized1=true;
//       // if (depth1>depth2)
//       //   std::cout << "bottom" << std::endl;
//     }
//   }
//
//
//   if (left2)
//   {
//     float deltax2 = -uv2.x()+(pixel_width/2);
//
//     float v=uv2.y()+steepness*deltax2;
//     if (v>=0 && v<=height)
//     {
//       float ratio_x = deltax2/delta_x;
//       float invdepth = ratio_x*invdepth_delta;
//       depth2 = 1.0/((1.0/depth2)+invdepth);
//       uv2.x()=(pixel_width/2);
//       uv2.y()=v;
//       resized2=true;
//     }
//   }
//   if (top2 && !resized2)
//   {
//     float deltay2=-uv2.y()+(pixel_width/2);
//
//     float u=uv2.x()+(1.0/steepness)*deltay2;
//     if (u>=0 && u<=width_)
//     {
//       float ratio_y = deltay2/delta_y;
//       float invdepth = ratio_y*invdepth_delta;
//       depth2 = 1.0/((1.0/depth2)+invdepth);
//       uv2.x()=u;
//       uv2.y()=(pixel_width/2);
//       resized2=true;
//     }
//   }
//   if (right2 && !resized2)
//   {
//     float deltax2 = width_-(pixel_width/2)-uv2.x();
//
//     float v=uv2.y()+steepness*deltax2;
//     if (v>=0 && v<=height)
//     {
//       float ratio_x = deltax2/delta_x;
//       float invdepth = ratio_x*invdepth_delta;
//       depth2 = 1.0/((1.0/depth2)+invdepth);
//       uv2.x()=width_-(pixel_width/2);
//       uv2.y()=v;
//       resized2=true;
//     }
//   }
//   if (bottom2 && !resized2)
//   {
//     float deltay2=height-(pixel_width/2)-uv2.y();
//
//     float u=uv2.x()+(1.0/steepness)*deltay2;
//     if (u>=0 && u<=width_)
//     {
//       float ratio_y = deltay2/delta_y;
//       float invdepth = ratio_y*invdepth_delta;
//       depth2 = 1.0/((1.0/depth2)+invdepth);
//       uv2.x()=u;
//       uv2.y()=height-(pixel_width/2);
//       resized2=true;
//     }
//   }
//   return true;
// }
