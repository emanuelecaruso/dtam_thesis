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

    bool extractCameraMatrix(Eigen::Matrix3f& K);
    void showWorldFrame(Eigen::Vector3f origin, float delta,int length);

    void clearImgs();
    void initImgs();

    void pixelCoords2uv(Eigen::Vector2i& pixel_coords, Eigen::Vector2f& uv);
    void uv2pixelCoords( Eigen::Vector2f& uv, Eigen::Vector2i& pixel_coords);

    bool pointAtDepth(Eigen::Vector2f& uv, float depth, Eigen::Vector3f& p);
    bool projectPoint(Eigen::Vector3f& p, Eigen::Vector2f& uv );
    bool projectPixel(Cp& p);
    void projectPixels(cpVector& cp_vector);
    void projectPixels_parallell(cpVector& cp_vector);


    inline Camera* clone(){
      return new Camera(*this);
    }

};

typedef std::vector<Camera*> CameraVector;
