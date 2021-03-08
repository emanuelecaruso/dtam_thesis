#pragma once
#include "camera.h"
#include "image.h"

class Dtam{
  public:
    CameraVector camera_vector_;
    int index_r_;
    Image<int>* cost_matrix_;
    Image<int>* n_valid_proj_matrix_;

    Dtam(){
      cost_matrix_ = new Image<int>("cost matrix");
      n_valid_proj_matrix_ = new Image<int>("matrix of valid projections");
    }

    void loadCameras(CameraVector camera_vector);
    void addCamera(Camera* camera);
    bool setReferenceCamera(int index_r);

    // bool get1stDepthWithUV(Camera* camera_r, Camera* camera_m, Eigen::Vector2f& uv_r, Eigen::Vector2f& uv_m, float& depth);
    void getDepthMap(int num_interpolations, bool check=false);
};
