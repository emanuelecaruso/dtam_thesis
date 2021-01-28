#include "camera.h"

using namespace pr;

bool Camera::extractCameraMatrix(Eigen::Matrix3f& K){
  float height= width_/aspect_;

  K <<
    lens_, 0    , width_/2 ,
    0    , lens_, height/2,
    0    ,   0  , 1       ;

  return true;
}

void Camera::clearImgs(){
  int cols = depth_map_->image_.cols;
  int rows = depth_map_->image_.rows;
  cv::Mat_< cv::Vec3b > image_rgb(cols,rows);
  image_rgb=cv::Vec3b(255,255,255);
  cv::Mat_< uchar > depth_map(cols,rows);
  depth_map=uchar(255);
  depth_map_->image_=depth_map;
  image_rgb_->image_=image_rgb;
}

void Camera::initImgs(){
  Image< uchar >* depth_map_img = new Image< uchar >("Depth map "+name_);
  cv::Mat_< uchar > depth_map(resolution_,(int)(resolution_/aspect_));
  depth_map=uchar(255);
  depth_map_img->image_=depth_map;
  depth_map_=depth_map_img;

  Image< cv::Vec3b >* rgb_image_img = new Image< cv::Vec3b >("rgb image "+name_);
  cv::Mat_< cv::Vec3b > rgb_image(resolution_,(int)(resolution_/aspect_));
  rgb_image=cv::Vec3b(255,255,255);
  rgb_image_img->image_=rgb_image;
  image_rgb_=rgb_image_img;
}


bool Camera::projectCp(Cp& cp){

  Eigen::Matrix3f K;
  Camera::extractCameraMatrix(K);

  Eigen::Vector3f p_cam = frame_world_wrt_camera*cp.point;

  if (p_cam.z()<=lens_)
    return false;

  Eigen::Vector3f p_proj = K*p_cam;

  Eigen::Vector2f uv = p_proj.head<2>()*(1./p_proj.z());

  if(uv.x()<0 || uv.x()>width_)
    return false;
  if(uv.y()<0 || uv.y()>width_/aspect_)
    return false;

  Eigen::Vector2i uv_int;
  uv_int.x()=(int)round((uv.x()/width_)*resolution_);
  uv_int.y()=(int)round((uv.y()/(width_/aspect_))*resolution_);


  int numb = (p_cam.z())*(255.0/2.0);
  uchar depth = numb;

  uchar evalued_pixel;
  depth_map_->evalPixel(uv_int,evalued_pixel);

  if (evalued_pixel<depth)
    return false;

  if (depth>255 || cp.color[0]>255 || cp.color[1]>255 || cp.color[2]>255)
    return false;


  image_rgb_->setPixel(uv_int, cp.color);
  depth_map_->setPixel(uv_int,depth);

  return true;
}

void Camera::projectCps(cpVector& cp_vector){
  Camera::clearImgs();
  for (Cp cp : cp_vector)
  {
    Camera::projectCp(cp);
  }
}
