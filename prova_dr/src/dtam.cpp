#include "dtam.h"

float Dtam::getSteepness(Eigen::Vector2f& uv1 ,Eigen::Vector2f& uv2 ){
  float delta_x = uv2.x()-uv1.x();
  float delta_y = uv2.y()-uv1.y();
  float steepness=delta_y/delta_x;
  return steepness;
}

void Dtam::resizeLine(Eigen::Vector2f& uv1 ,Eigen::Vector2f& uv2, float width, float height, float resolution ){

  float pixel_width= width/resolution;
  float steepness = Dtam::getSteepness(uv1 ,uv2 );
  bool done1 = false; bool done2 = false;
  bool top1 = false; bool bottom1 = false; bool left1 = false; bool right1 = false;
  bool top2 = false; bool bottom2 = false; bool left2 = false; bool right2 = false;
  if ( uv1.x() < 0 )
    left1=true;
  else if (uv1.x() > width)
    right1=true;
  if ( uv1.y() < 0 )
    top1=true;
  else if (uv1.y() > height)
    bottom1=true;
  if ( uv2.x() < 0 )
    left2=true;
  else if (uv2.x() > width)
    right2=true;
  if ( uv2.y() < 0 )
    top2=true;
  else if (uv2.y() > height)
    bottom2=true;

    if (left1)
    {
      float v=uv1.y()+steepness*(-uv1.x()+(pixel_width/2));
      if (v>=0 && v<=height)
      {
        uv1.x()=(pixel_width/2);
        uv1.y()=v;
        done1= true;
      }
    }
    if (top1 && !done1)
    {
      float u=uv1.x()+(1/steepness)*(-uv1.y()+(pixel_width/2));
      if (u>=0 && u<=width)
      {
        uv1.x()=u;
        uv1.y()=0;
        done1= true;
      }
    }
    if (right1 && !done1)
    {
      float v=uv1.y()+steepness*(width-(pixel_width/2)-uv1.x());
      if (v>=0 && v<=height)
      {
        uv1.x()=width-(pixel_width/2);
        uv1.y()=v;
        done1= true;
      }
    }
    if (bottom1 && !done1)
    {
      float u=uv1.x()+(1/steepness)*(height-(pixel_width/2)-uv1.y());
      if (u>=0 && u<=width)
      {
        uv1.x()=u;
        uv1.y()=height-(pixel_width/2);
        done1= true;
      }
    }


    if (left2)
    {
      float v=uv2.y()+steepness*(-uv2.x()+(pixel_width/2));
      if (v>=0 && v<=height)
      {
        uv2.x()=(pixel_width/2);
        uv2.y()=v;
        done2= true;
      }
    }
    if (top2 && !done2)
    {
      float u=uv2.x()+(1/steepness)*(-uv2.y()+(pixel_width/2));

      if (u>=0 && u<=width)
      {
        uv2.x()=u;
        uv2.y()=(pixel_width/2);
        done2= true;
      }
    }
    if (right2 && !done2)
    {
      float v=uv2.y()+steepness*(width-(pixel_width/2)-uv2.x());

      if (v>=0 && v<=height)
      {
        uv2.x()=width-(pixel_width/2);
        uv2.y()=v;
        done2= true;
      }
    }
    if (bottom2 && !done2)
    {
      float u=uv2.x()+(1/steepness)*(height-(pixel_width/2)-uv2.y());

      if (u>=0 && u<=width)
      {
        uv2.x()=u;
        uv2.y()=height-(pixel_width/2);
        done2= true;
      }
    }

}

bool Dtam::get1stDepthWithUV(Camera* camera_r, Camera* camera_m, Eigen::Vector2f& uv_r, Eigen::Vector2f& uv_m, float& depth){

  Eigen::Isometry3f T = camera_m->frame_world_wrt_camera_*camera_r->frame_camera_wrt_world_;
  auto r=T.linear();
  auto t=T.translation();
  float f = camera_r->lens_;
  float w=camera_m->width_;
  float h=camera_m->width_/camera_m->aspect_;
  depth = (2*f*(f+t(2)))/(2*f*r(2,2)-2*r(2,0)*uv_r.x()+r(2,0)*w-r(2,1)*(h-2*uv_r.y()));
  uv_m.x() = t(0)+(w/2)-depth*r(0,2)+((depth*r(0,0)*(2*uv_r.x()-w))/(2*f))+((depth*r(0,1)*(h-2*uv_r.y()))/(2*f));
  uv_m.y() = (h/2)-t(1)+depth*r(1,2)-((depth*r(1,0)*(2*uv_r.x()-w))/(2*f))-((depth*r(1,1)*(h-2*uv_r.y()))/(2*f));

}


bool Dtam::getEpipolarLine(Eigen::Vector2i& pixel_coords_r, Camera* camera_r, Camera* camera_m)
{
  Eigen::Vector3f camera_r_p = camera_r->frame_camera_wrt_world_.translation();
  Eigen::Vector2f cam_r_projected_on_cam_m;
  float cam_r_depth_on_camera_m;
  bool cam_r_in_front = camera_m->projectPoint(camera_r_p, cam_r_projected_on_cam_m, cam_r_depth_on_camera_m);

  Eigen::Vector3f query_p = camera_r->frame_camera_wrt_world_.translation();
  Eigen::Vector2f uv_r;
  camera_r->pixelCoords2uv(pixel_coords_r, uv_r);
  camera_r->pointAtDepth(uv_r, camera_r->max_depth_, query_p);
  // camera_r->pointAtDepth(uv_r, 0.1351, query_p);
  Eigen::Vector2f query_p_projected_on_cam_m;
  float query_depth_on_camera_m;
  bool query_in_front = camera_m->projectPoint(query_p, query_p_projected_on_cam_m, query_depth_on_camera_m);

  // std::cout << "query p:\n" << query_p << std::endl;
  // Eigen::Vector3f p_cam = camera_m->frame_world_wrt_camera_*query_p;
  //
  // std::cout << "\nquery p on cam m :\n" << p_cam << std::endl;
  // std::cout << query_depth_on_camera_m << std::endl;

  Eigen::Vector2f uv1;
  Eigen::Vector2f uv2;

  // if both camera r and query point are on back of camera m return false
  if (!query_in_front && !cam_r_in_front)
    return false;
  // if query point is in front of camera m whereas camera r is on the back
  else if (query_in_front && !cam_r_in_front){

    uv2=query_p_projected_on_cam_m;
    float depth_m;
    Dtam::get1stDepthWithUV(camera_r, camera_m, uv_r, uv1, depth_m);
    resizeLine(uv1 , uv2, camera_m->width_, camera_m->width_/camera_m->aspect_, camera_m->resolution_ );
  }
  // if camera r is in front of camera m whereas query point is on the back
  else if (!query_in_front && cam_r_in_front){
    // TODO
    return false;
  }
  // if both camera r and query point are in front of camera m
  else {
    uv1=cam_r_projected_on_cam_m;
    uv2=query_p_projected_on_cam_m;
    resizeLine(uv1 , uv2, camera_m->width_, camera_m->width_/camera_m->aspect_, camera_m->resolution_ );
  }

  // std::cout << cam_r_projected_on_cam_m << std::endl;

  Eigen::Vector2i pixel_coords_1;
  Eigen::Vector2i pixel_coords_2;

  camera_r->uv2pixelCoords( uv1, pixel_coords_1);
  camera_r->uv2pixelCoords( uv2, pixel_coords_2);

  cv::Vec3b clr1(0,0,255);
  cv::Vec3b clr2(0,255,0);
  camera_m->image_rgb_->setPixel(pixel_coords_1, clr1);
  camera_m->image_rgb_->setPixel(pixel_coords_2, clr2);

  // std::cout << "width: "<< camera_m->width_ << std::endl;
  // std::cout << "pixel width/2: "<< (camera_m->width_/camera_m->resolution_)/2 << std::endl;
  // std::cout << uv1 << std::endl;
  // std::cout << uv2 << std::endl;
  // std::cout << pixel_coords_1 << std::endl;
  // std::cout << pixel_coords_2 << std::endl;


  // camera_m_->projectPixel(cp);
  camera_m->image_rgb_->show(800/camera_m->resolution_);

  return true;
}
