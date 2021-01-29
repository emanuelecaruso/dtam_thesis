#include "dtam.h"

bool Dtam::sign_epipolar_line(Eigen::Vector2i& uv, Camera* camera_r, Camera* camera_m)
{
  Camera* camera_r_=camera_r->clone();
  Camera* camera_m_=camera_m->clone();
  camera_r_->clearImgs();
  camera_m_->clearImgs();
  Cp cp;
  cp.point = camera_r_->frame_camera_wrt_world_.translation();
  cp.point.z()+= -0.5;
  cp.color = cv::Vec3b(0,255,255);
  camera_m_->projectCp(cp);
  camera_m_->image_rgb_->show(800/camera_m_->resolution_);

  return true;
}
