#pragma once
#include "defs.h"
#include "image.h"

using namespace pr;

class Camera{
  public:
    std::string name_;
    float lens_;
    float aspect_;
    float width_;
    int resolution_;
    Eigen::Isometry3f frame_world_wrt_camera;
    Image<uchar>* depth_map_;
    Image<cv::Vec3b>* image_rgb_;

    Camera(std::string name, float lens,
           float aspect, float width, int resolution, Eigen::Isometry3f frame){
       name_ = name;
       lens_ = lens;
       aspect_ = aspect;
       width_ = width;
       resolution_ = resolution;
       frame_world_wrt_camera = frame;
    };


    void clearImgs();
    void initImgs();
    bool projectCp(Cp& p);
    void projectCps(cpVector& cp_vector);

  private:
    bool extractCameraMatrix(Eigen::Matrix3f& K);
    // bool cuv2cp(Cuv& cuv, Cp& cp);
};

typedef std::vector<Camera*> CameraVector;
