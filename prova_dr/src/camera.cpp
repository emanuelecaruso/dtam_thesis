#include "camera.h"
#include <thread>
#include <vector>
#include <mutex>

using namespace std;
using namespace pr;



void Camera::clearImgs(){
  depth_map_->setAllPixels(1.0);
  image_rgb_->setAllPixels(cv::Vec3b(255,255,255));
}

void Camera::printMembers(){

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


void Camera::pixelCoords2uv(Eigen::Vector2i& pixel_coords, Eigen::Vector2f& uv){
  float pixel_width = width_/resolution_;

  uv.x()=((float)pixel_coords.x()/(resolution_))*width_+(pixel_width/2);
  uv.y()=(((float)pixel_coords.y())/(float)((resolution_)/aspect_))*(float)(width_/aspect_)+(pixel_width/2);
}

void Camera::uv2pixelCoords( Eigen::Vector2f& uv, Eigen::Vector2i& pixel_coords){

  pixel_coords.x()=(int)((uv.x()/width_)*resolution_);
  pixel_coords.y()=(int)((uv.y()/(width_/aspect_))*(resolution_/aspect_));
}

void Camera::pointAtDepth(Eigen::Vector2f& uv, float depth, Eigen::Vector3f& p){

  Eigen::Vector3f p_proj;
  Eigen::Vector2f product = uv * depth;
  p_proj.x() = product.x();
  p_proj.y() = product.y();
  p_proj.z() = depth;
  Eigen::Vector3f p_cam = Kinv_*p_proj;
  p = *frame_camera_wrt_world_*p_cam;

}

bool Camera::projectPoint(Eigen::Vector3f& p, Eigen::Vector2f& uv, float& p_cam_z ){


  Eigen::Vector3f p_cam = *frame_world_wrt_camera_*p;

  // return wether the projected point is in front or behind the camera
  p_cam_z=-p_cam.z();

  Eigen::Vector3f p_proj = K_*p_cam;

  uv = p_proj.head<2>()*(1./p_proj.z());

  if (p_cam_z<lens_)
    return false;

  return true;

}

void Camera::saveRGB(std::string path){
  cv::imwrite(path+ "/rgb_" +name_+".png", image_rgb_->image_);
}

void Camera::saveDepthMap(std::string path){
  cv::Mat ucharImg;
  depth_map_->image_.convertTo(ucharImg, CV_32FC1, 255.0);
  cv::imwrite(path+ "/depth_" +name_+".png", ucharImg);

}

void Camera::loadRGB(std::string path){

  image_rgb_->image_=cv::imread(path);

}

void Camera::loadDepthMap(std::string path){
  cv::Mat_<cv::Vec3b> rgbImg;
  cv::Mat channel[3];
  rgbImg = cv::imread(path);
  split(rgbImg, channel);
  depth_map_->image_=channel[0];
  depth_map_->image_.convertTo(depth_map_->image_, CV_32FC1, 1.0/255.0);
}

void Camera::showWorldFrame(Eigen::Vector3f origin, float delta, int length){
  Camera::clearImgs();
  std::vector<Cp> cps_world_frame;
  for (int i=0; i<length; i++)
  {
    Eigen::Vector3f x(origin[0]+delta*i,origin[1],origin[2]);
    cv::Vec3b color1(0,0,255);
    struct Cp cp1 = {x, color1};
    cps_world_frame.push_back(cp1);

    Eigen::Vector3f y(origin[0],origin[1]+delta*i,origin[2]);
    cv::Vec3b color2(0,255,0);
    struct Cp cp2 = {y, color2};
    cps_world_frame.push_back(cp2);

    if (i>0)
    {
      Eigen::Vector3f z(origin[0],origin[1],origin[2]+delta*i);
      cv::Vec3b color3(255,0,0);
      struct Cp cp3 = {z, color3};
      cps_world_frame.push_back(cp3);
    }
  }


  int cols=resolution_;
  int rows=resolution_/aspect_;

  for (Cp cp : cps_world_frame){

    Eigen::Vector2f uv;
    float p_cam_z;

    if (!projectPoint(cp.point, uv, p_cam_z ))
      continue;

    Eigen::Vector2i pixel_coords;
    Camera::uv2pixelCoords( uv, pixel_coords);

    if (cp.color[0]>255 || cp.color[1]>255 || cp.color[2]>255)
      continue;

    int r=pixel_coords.y();
    int c=pixel_coords.x();
    if(r<0||r>=rows)
      continue;
    if(c<0||c>=cols)
      continue;

    cv::circle(image_rgb_->image_, cv::Point(c,r), 3, cp.color);
  }

}
