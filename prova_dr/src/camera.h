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
    float max_depth_;
    Eigen::Isometry3f frame_camera_wrt_world_;
    Eigen::Isometry3f frame_world_wrt_camera_;
    Image<float>* depth_map_;
    Image<cv::Vec3b>* image_rgb_;

    Camera(std::string name, float lens, float aspect, float width, int resolution,
       float max_depth, Eigen::Isometry3f frame_camera_wrt_world, Eigen::Isometry3f frame_world_wrt_camera){
       name_ = name;
       lens_ = lens;
       aspect_ = aspect;
       width_ = width;
       resolution_ = resolution;
       max_depth_ = max_depth;
       frame_camera_wrt_world_ = frame_camera_wrt_world;
       frame_world_wrt_camera_ = frame_world_wrt_camera;
    };


    void showWorldFrame(Eigen::Vector3f origin, float delta,int length);
    void clearImgs();
    void initImgs();
    bool projectCp(Cp& p);
    void projectCps(cpVector& cp_vector);
    void projectCps_parallell(cpVector& cp_vector);
    void test();

    inline Camera* clone(){
      return new Camera(*this);
    }

  private:
    bool extractCameraMatrix(Eigen::Matrix3f& K);
    // bool cuv2cp(Cuv& cuv, Cp& cp);
};

typedef std::vector<Camera*> CameraVector;
