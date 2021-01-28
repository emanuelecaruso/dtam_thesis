#pragma once

class Dtam{
  public:
    CameraVector camera_vector_;
    Class(CameraVector camera_vector){
      camera_vector_ = camera_vector;
    };

    bool get_d(Eigen::Vector2i& uv, Camera* camera_r, Camera* camera_m, std::vector<floatZ d);
};
