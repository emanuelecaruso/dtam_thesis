#include "camera.h"
#include <thread>
#include <vector>
#include <mutex>

using namespace std;
using namespace pr;

bool Camera::extractCameraMatrix(Eigen::Matrix3f& K){
  float height= width_/aspect_;

  K <<
    lens_,   0   ,  -width_/2 ,
    0    ,  -lens_, -height/2,
    0    ,   0   ,   -1       ;

  return true;
}


void Camera::clearImgs(){
  depth_map_->image_=1.0;
  image_rgb_->image_=cv::Vec3b(255,255,255);
}

void Camera::initImgs(){
  Image< float >* depth_map_img = new Image< float >("Depth map "+name_);
  cv::Mat_< float > depth_map((int)(resolution_/aspect_),resolution_);
  depth_map=1.0;
  depth_map_img->image_=depth_map;
  depth_map_=depth_map_img;

  Image< cv::Vec3b >* rgb_image_img = new Image< cv::Vec3b >("rgb image "+name_);
  cv::Mat_< cv::Vec3b > rgb_image((int)(resolution_/aspect_),resolution_);
  rgb_image=cv::Vec3b(255,255,255);
  rgb_image_img->image_=rgb_image;
  image_rgb_=rgb_image_img;
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

  Eigen::Matrix3f K;
  Camera::extractCameraMatrix(K);

  int cols=resolution_;
  int rows=resolution_/aspect_;

  for (Cp cp : cps_world_frame){
    Eigen::Vector3f p_cam = frame_world_wrt_camera_*cp.point;

    if (p_cam.z()>-lens_)
      continue;

    Eigen::Vector3f p_proj = K*p_cam;

    Eigen::Vector2f uv = p_proj.head<2>()*(1./p_proj.z());

    if(uv.x()<0 || uv.x()>width_)
      continue;
    if(uv.y()<0 || uv.y()>(width_/aspect_))
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

  Eigen::Matrix3f K;
  Camera::extractCameraMatrix(K);

  Eigen::Vector3f p_proj;
  Eigen::Vector2f product = uv * depth;
  p_proj.x() = product.x();
  p_proj.y() = product.y();
  p_proj.z() = depth;
  Eigen::Vector3f p_cam = K.inverse()*p_proj;
  p = frame_camera_wrt_world_*p_cam;

}

bool Camera::projectPoint(Eigen::Vector3f& p, Eigen::Vector2f& uv, float& p_cam_z ){

  Eigen::Matrix3f K;
  Camera::extractCameraMatrix(K);

  Eigen::Vector3f p_cam = frame_world_wrt_camera_*p;

  // return wether the projected point is in front or behind the camera
  p_cam_z=-p_cam.z();
  if (p_cam_z<lens_)
    return false;

  Eigen::Vector3f p_proj = K*p_cam;

  uv = p_proj.head<2>()*(1./p_proj.z());


  return true;

}

bool Camera::projectPixel(Cp& cp){

  Eigen::Vector2f uv;
  float depth_cam;
  bool point_in_front_of_camera = Camera::projectPoint(cp.point, uv, depth_cam );
  if (!point_in_front_of_camera)
    return false;

  if(uv.x()<0 || uv.x()>width_)
    return false;
  if(uv.y()<0 || uv.y()>width_/aspect_)
    return false;

  Eigen::Vector2i pixel_coords;
  Camera::uv2pixelCoords( uv, pixel_coords);

  float depth = depth_cam/max_depth_;

  float evaluated_pixel;
  depth_map_->evalPixel(pixel_coords,evaluated_pixel);

  if (evaluated_pixel<depth)
    return false;

  if (depth>1 || depth>255 || cp.color[0]>255 || cp.color[1]>255 || cp.color[2]>255)
    return false;


  image_rgb_->setPixel(pixel_coords, cp.color);
  depth_map_->setPixel(pixel_coords,depth);

  return true;
}

void Camera::projectPixels(cpVector& cp_vector){
  Camera::clearImgs();
  for (Cp cp : cp_vector)
  {
    Camera::projectPixel(cp);
  }
}


void Camera::projectPixels_parallell(cpVector& cp_vector){

  Camera::clearImgs();
  const size_t nloop = cp_vector.size();
  const size_t nthreads = std::thread::hardware_concurrency();
  {
    // Pre loop
    std::vector<std::thread> threads(nthreads);
    std::mutex critical;
    for(int t = 0;t<nthreads;t++)
    {
      threads[t] = std::thread(std::bind(
        [&](const int bi, const int ei, const int t)
        {
          // loop over all items
          for(int i = bi;i<ei;i++)
          {
            // inner loop
            {
              Camera::projectPixel(cp_vector[i]);
            }
          }
        },t*nloop/nthreads,(t+1)==nthreads?nloop:(t+1)*nloop/nthreads,t));
    }
    std::for_each(threads.begin(),threads.end(),[](std::thread& x){x.join();});
    // Post loop
  }

}

void Camera::resizeLine(Eigen::Vector2f& uv1 ,Eigen::Vector2f& uv2, float& depth1, float& depth2){

  float pixel_width= width_/resolution_;
  float height = width_/aspect_;

  float delta_x = uv2.x()-uv1.x();
  float delta_y = uv2.y()-uv1.y();
  float steepness=delta_y/delta_x;
  float delta_depth=depth2-depth1;

  bool done1 = false; bool done2 = false;
  bool top1 = false; bool bottom1 = false; bool left1 = false; bool right1 = false;
  bool top2 = false; bool bottom2 = false; bool left2 = false; bool right2 = false;
  if ( uv1.x() < 0 )
    left1=true;
  else if (uv1.x() > width_)
    right1=true;
  if ( uv1.y() < 0 )
    top1=true;
  else if (uv1.y() > height)
    bottom1=true;
  if ( uv2.x() < 0 )
    left2=true;
  else if (uv2.x() > width_)
    right2=true;
  if ( uv2.y() < 0 )
    top2=true;
  else if (uv2.y() > height)
    bottom2=true;

  if (left1)
  {
    float deltax1 = -uv1.x()+(pixel_width/2);
    float v=uv1.y()+steepness*deltax1;
    float ratio_x = deltax1/delta_x;
    depth1 = delta_depth*ratio_x;
    if (v>=0 && v<=height)
    {
      uv1.x()=(pixel_width/2);
      uv1.y()=v;
      done1= true;
    }
  }
  if (top1 && !done1)
  {
    float deltay1=-uv1.y()+(pixel_width/2);
    float u=uv1.x()+(1/steepness)*deltay1;
    float ratio_y = deltay1/delta_y;
    depth1 = delta_depth*ratio_y;
    if (u>=0 && u<=width_)
    {
      uv1.x()=u;
      uv1.y()=0;
      done1= true;
    }
  }
  if (right1 && !done1)
  {
    float deltax1 = width_-(pixel_width/2)-uv1.x();
    float v=uv1.y()+steepness*deltax1;
    float ratio_x = deltax1/delta_x;
    depth1 = delta_depth*ratio_x;
    if (v>=0 && v<=height)
    {
      uv1.x()=width_-(pixel_width/2);
      uv1.y()=v;
      done1= true;
    }
  }
  if (bottom1 && !done1)
  {
    float deltay1=height-(pixel_width/2)-uv1.y();
    float u=uv1.x()+(1/steepness)*deltay1;
    float ratio_y = deltay1/delta_y;
    depth1 = delta_depth*ratio_y;
    if (u>=0 && u<=width_)
    {
      uv1.x()=u;
      uv1.y()=height-(pixel_width/2);
      done1= true;
    }
  }


  if (left2)
  {
    float deltax2 = -uv2.x()+(pixel_width/2);
    float v=uv2.y()+steepness*deltax2;
    float ratio_x = deltax2/delta_x;
    depth2 = delta_depth*ratio_x;
    if (v>=0 && v<=height)
    {
      uv2.x()=(pixel_width/2);
      uv2.y()=v;
      done2= true;
    }
  }
  if (top2 && !done2)
  {
    float deltay2=-uv2.y()+(pixel_width/2);
    float u=uv2.x()+(1/steepness)*deltay2;
    float ratio_y = deltay2/delta_y;
    depth2 = delta_depth*ratio_y;
    if (u>=0 && u<=width_)
    {
      uv2.x()=u;
      uv2.y()=(pixel_width/2);
      done2= true;
    }
  }
  if (right2 && !done2)
  {
    float deltax2 = width_-(pixel_width/2)-uv2.x();
    float v=uv2.y()+steepness*deltax2;
    float ratio_x = deltax2/delta_x;
    depth2 = delta_depth*ratio_x;
    if (v>=0 && v<=height)
    {
      uv2.x()=width_-(pixel_width/2);
      uv2.y()=v;
      done2= true;
    }
  }
  if (bottom2 && !done2)
  {
    float deltay2=height-(pixel_width/2)-uv2.y();
    float u=uv2.x()+(1/steepness)*deltay2;
    float ratio_y = deltay2/delta_y;
    depth2 = delta_depth*ratio_y;
    if (u>=0 && u<=width_)
    {
      uv2.x()=u;
      uv2.y()=height-(pixel_width/2);
      done2= true;
    }
  }

}
