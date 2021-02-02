#include "dtam.h"


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

  Eigen::Vector2f uv1;
  Eigen::Vector2f uv2;
  float depth1;
  float depth2;

  // if both camera r and query point are on back of camera m return false
  if (!query_in_front && !cam_r_in_front)
    return false;
  // if query point is in front of camera m whereas camera r is on the back
  else if (query_in_front && !cam_r_in_front){
    std::cout << "query in front" << std::endl;

    uv2=query_p_projected_on_cam_m;
    depth2=query_depth_on_camera_m;
    Dtam::get1stDepthWithUV(camera_r, camera_m, uv_r, uv1, depth1);
    camera_m->resizeLine(uv1 , uv2, depth1, depth2 );

  }
  // if camera r is in front of camera m whereas query point is on the back
  else if (!query_in_front && cam_r_in_front){
    // TODO
    return false;
  }
  // if both camera r and query point are in front of camera m
  else {
    std::cout << "both in front" << std::endl;
    uv1=cam_r_projected_on_cam_m;
    uv2=query_p_projected_on_cam_m;
    depth1=cam_r_depth_on_camera_m;
    depth2=query_depth_on_camera_m;

    // camera_m->resizeLine(uv1 , uv2, depth1, depth2);
    // std::cout << "Depth1 new: " << depth1 << std::endl;
    // std::cout << "Depth2 new: " << depth2 << "\n\n" << std::endl;

    Eigen::Vector3f pt;
    // camera_m->pointAtDepth(uv2, depth2, pt);
    camera_m->pointAtDepth(uv2, depth2, pt);

    Eigen::Vector2f uv_;
    float d;
    camera_r->projectPoint(pt,uv_, d );
    Eigen::Vector2i pxl;
    camera_r->uv2pixelCoords( uv_, pxl);

    std::cout << uv_ << std::endl;
    std::cout << pxl << std::endl;

  }

  Eigen::Vector2i pixel_coords_1;
  Eigen::Vector2i pixel_coords_2;

  camera_r->uv2pixelCoords( uv1, pixel_coords_1);
  camera_r->uv2pixelCoords( uv2, pixel_coords_2);

  cv::Vec3b clr1(0,0,255);
  cv::Vec3b clr2(0,255,0);
  camera_m->image_rgb_->setPixel(pixel_coords_1, clr1);
  camera_m->image_rgb_->setPixel(pixel_coords_2, clr2);



  // camera_m_->projectPixel(cp);
  camera_m->image_rgb_->show(800/camera_m->resolution_);

  return true;
}
