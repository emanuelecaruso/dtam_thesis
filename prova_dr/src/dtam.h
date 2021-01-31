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

    float getSteepness(Eigen::Vector2f& uv1 ,Eigen::Vector2f& uv2 );
    void resizeLine(Eigen::Vector2f& uv1 ,Eigen::Vector2f& uv2 , float width, float height, float resolution );
    bool getEpipolarLine(Eigen::Vector2i& uv_r, Camera* camera_r, Camera* camera_m);
};
