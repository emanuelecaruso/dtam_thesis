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
    float min_depth_;
    Eigen::Matrix3f K_;
    Eigen::Matrix3f Kinv_;
    Image<float>* invdepth_map_;
    Image<cv::Vec3b>* image_rgb_;
    Eigen::Isometry3f* frame_world_wrt_camera_;
    Eigen::Isometry3f* frame_camera_wrt_world_;

    Camera(std::string name, float lens, float aspect, float width, int resolution,
       float max_depth, float min_depth, Eigen::Isometry3f* frame_camera_wrt_world, Eigen::Isometry3f* frame_world_wrt_camera){
       name_ = name;
       lens_ = lens;
       aspect_ = aspect;
       width_ = width;
       resolution_ = resolution;
       max_depth_ = max_depth;
       min_depth_ = min_depth;
       frame_camera_wrt_world_ = frame_camera_wrt_world;
       frame_world_wrt_camera_ = frame_world_wrt_camera;
       invdepth_map_ = new Image< float >("invdepth_"+name_);
       image_rgb_ = new Image< cv::Vec3b >("rgb_"+name_);

       // initialize images with white color
       invdepth_map_->initImage(resolution_/aspect_,resolution_);
       invdepth_map_->setAllPixels(1.0);
       image_rgb_->initImage(resolution_/aspect_,resolution_);
       image_rgb_->setAllPixels(cv::Vec3b(255,255,255));

       // compute camera matrix and its inverse
       K_ <<
         lens_,   0   ,     -width_/2       ,
         0    ,  -lens_, -(width_/aspect_)/2,
         0    ,   0   ,       -1            ;
       Kinv_=K_.inverse();
    };

    void showWorldFrame(Eigen::Vector3f origin, float delta,int length);
    void printMembers();

    void clearImgs();

    void pixelCoords2uv(Eigen::Vector2i& pixel_coords, Eigen::Vector2f& uv);
    void uv2pixelCoords( Eigen::Vector2f& uv, Eigen::Vector2i& pixel_coords);

    void pointAtDepth(Eigen::Vector2f& uv, float depth, Eigen::Vector3f& p);
    bool projectPoint(Eigen::Vector3f& p, Eigen::Vector2f& uv, float& p_cam_z );

    void saveRGB(std::string path);
    void saveDepthMap(std::string path);

    void loadRGB(std::string path);
    void loadDepthMap(std::string path);

    inline Camera* clone(){
      return new Camera(*this);
    }

};

typedef std::vector<Camera*> CameraVector;
