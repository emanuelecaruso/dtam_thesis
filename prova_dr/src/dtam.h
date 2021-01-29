#pragma once
#include "camera.h"
#include "image.h"

class Dtam{
  public:
    int i_;
    Image<float>* img_cam_r_;
    Image<float>* img_cam_m_;
    Dtam(int i){
      i_ = i;
    };

    bool sign_epipolar_line(Eigen::Vector2i& uv, Camera* camera_r, Camera* camera_m);
};
