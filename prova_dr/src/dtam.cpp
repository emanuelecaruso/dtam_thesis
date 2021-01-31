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
  std::cout << "steepness: " << steepness << std::endl;
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

bool Dtam::getEpipolarLine(Eigen::Vector2i& pixel_coords_r, Camera* camera_r, Camera* camera_m)
{
  Eigen::Vector3f camera_r_p = camera_r->frame_camera_wrt_world_.translation();
  Eigen::Vector2f cam_r_projected_on_cam_m;
  camera_m->projectPoint(camera_r_p, cam_r_projected_on_cam_m);

  Eigen::Vector3f query_p = camera_r->frame_camera_wrt_world_.translation();
  Eigen::Vector2f uv_float;
  camera_r->pixelCoords2uv(pixel_coords_r, uv_float);
  // camera_r->pointAtDepth(uv_float, camera_r->max_depth_, query_p);
  camera_r->pointAtDepth(uv_float, 2, query_p);
  Eigen::Vector2f query_p_projected_on_cam_m;
  camera_m->projectPoint(query_p, query_p_projected_on_cam_m);

  std::cout << cam_r_projected_on_cam_m << std::endl;


  float steepness = getSteepness( cam_r_projected_on_cam_m , query_p_projected_on_cam_m );
  resizeLine(cam_r_projected_on_cam_m ,query_p_projected_on_cam_m, camera_r->width_, camera_r->width_/camera_r->aspect_, camera_r->resolution_ );

  Eigen::Vector2i pixel_coords_cam_r;
  Eigen::Vector2i pixel_coords_query;


  camera_r->uv2pixelCoords( cam_r_projected_on_cam_m, pixel_coords_cam_r);
  camera_r->uv2pixelCoords( query_p_projected_on_cam_m, pixel_coords_query);

  cv::Vec3b clr(0,0,255);
  camera_m->image_rgb_->setPixel(pixel_coords_cam_r, clr);
  camera_m->image_rgb_->setPixel(pixel_coords_query, clr);

  std::cout << "width: "<< camera_m->width_ << std::endl;
  std::cout << "pixel width/2: "<< (camera_m->width_/camera_m->resolution_)/2 << std::endl;
  // std::cout << pixel_coords_r << std::endl;
  std::cout << cam_r_projected_on_cam_m << std::endl;
  // std::cout << query_p_projected_on_cam_m << std::endl;
  // std::cout << pixel_coords_cam_r << std::endl;
  // std::cout << pixel_coords_query << std::endl;


  // camera_m_->projectPixel(cp);
  camera_m->image_rgb_->show(800/camera_m->resolution_);

  return true;
}
