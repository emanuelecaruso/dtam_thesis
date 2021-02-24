#pragma once
#include "camera.h"
#include "image.h"
#include <cuda_runtime.h>

class Dtam{
  public:
    CameraVector camera_vector_;
    Dtam(CameraVector camera_vector){
      camera_vector_ = camera_vector;
    };

    bool get1stDepthWithUV(Camera* camera_r, Camera* camera_m, Eigen::Vector2f& uv_r, Eigen::Vector2f& uv_m, float& depth);
    void getDepthMap(int num_interpolations, bool check=false);
};
