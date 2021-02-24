#pragma once
#include "defs.h"
#include "image.h"

using namespace pr;

#ifndef USE_CUDA

class Camera{
  public:

    std::string name_;
    float lens_;
    float aspect_;
    float width_;
    int resolution_;
    float max_depth_;
    Image<float>* depth_map_;
    Image<cv::Vec3b>* image_rgb_;
    Eigen::Isometry3f* frame_world_wrt_camera_;
    Eigen::Isometry3f* frame_camera_wrt_world_;

    Camera(std::string name, float lens, float aspect, float width, int resolution,
       float max_depth, Eigen::Isometry3f* frame_camera_wrt_world, Eigen::Isometry3f* frame_world_wrt_camera){
       name_ = name;
       lens_ = lens;
       aspect_ = aspect;
       width_ = width;
       resolution_ = resolution;
       max_depth_ = max_depth;
       frame_camera_wrt_world_ = frame_camera_wrt_world;
       frame_world_wrt_camera_ = frame_world_wrt_camera;
       depth_map_ = new Image< float >("Depth map "+name_);
       image_rgb_ = new Image< cv::Vec3b >("rgb image "+name_);

       // initialize images with white color
       cv::Mat_< float > depth_map((int)(resolution_/aspect_),resolution_);
       depth_map=1.0;
       depth_map_->image_=depth_map;
       cv::Mat_< cv::Vec3b > rgb_image((int)(resolution_/aspect_),resolution_);
       rgb_image=cv::Vec3b(255,255,255);
       image_rgb_->image_=rgb_image;
    };

    bool extractCameraMatrix(Eigen::Matrix3f& K);
    void showWorldFrame(Eigen::Vector3f origin, float delta,int length);

    void clearImgs();

    void pixelCoords2uv(Eigen::Vector2i& pixel_coords, Eigen::Vector2f& uv);
    void uv2pixelCoords( Eigen::Vector2f& uv, Eigen::Vector2i& pixel_coords);

    void pointAtDepth(Eigen::Vector2f& uv, float depth, Eigen::Vector3f& p);
    bool projectPoint(Eigen::Vector3f& p, Eigen::Vector2f& uv, float& p_cam_z );
    bool projectPixel(Cp& p);
    void projectPixels(cpVector& cp_vector);
    void projectPixels_parallell(cpVector& cp_vector);

    bool resizeLine(Eigen::Vector2f& uv1 ,Eigen::Vector2f& uv2, float& depth1, float& depth2, bool& resized1, bool& resized2);

    inline Camera* clone(){
      return new Camera(*this);
    }

};

typedef std::vector<Camera*> CameraVector;

#else

class Camera{
  public:

    std::string name_;
    float lens_;
    float aspect_;
    float width_;
    int resolution_;
    float max_depth_;
    Image<float>* depth_map_;
    Image<cv::Vec3b>* image_rgb_;
    Eigen::Isometry3f* frame_world_wrt_camera_;
    Eigen::Isometry3f* frame_camera_wrt_world_;
    cv::cuda::GpuMat depth_map_gpu_;
    cv::cuda::GpuMat image_rgb_gpu_;

    Camera(std::string name, float lens, float aspect, float width, int resolution,
       float max_depth, Eigen::Isometry3f* frame_camera_wrt_world, Eigen::Isometry3f* frame_world_wrt_camera){
       name_ = name;
       lens_ = lens;
       aspect_ = aspect;
       width_ = width;
       resolution_ = resolution;
       max_depth_ = max_depth;
       frame_camera_wrt_world_ = frame_camera_wrt_world;
       frame_world_wrt_camera_ = frame_world_wrt_camera;
       depth_map_ = new Image< float >("Depth map "+name_);
       image_rgb_ = new Image< cv::Vec3b >("rgb image "+name_);

       // initialize images with white color
       cv::Mat_< float > depth_map((int)(resolution_/aspect_),resolution_);
       depth_map=1.0;
       depth_map_->image_=depth_map;
       cv::Mat_< cv::Vec3b > rgb_image((int)(resolution_/aspect_),resolution_);
       rgb_image=cv::Vec3b(255,255,255);
       image_rgb_->image_=rgb_image;
    };

    bool extractCameraMatrix(Eigen::Matrix3f& K);
    void showWorldFrame(Eigen::Vector3f origin, float delta,int length);

    void clearImgs();

    void pixelCoords2uv(Eigen::Vector2i& pixel_coords, Eigen::Vector2f& uv);
    void uv2pixelCoords( Eigen::Vector2f& uv, Eigen::Vector2i& pixel_coords);

    void pointAtDepth(Eigen::Vector2f& uv, float depth, Eigen::Vector3f& p);
    bool projectPoint(Eigen::Vector3f& p, Eigen::Vector2f& uv, float& p_cam_z );
    bool projectPixel(Cp& p);
    void projectPixels(cpVector& cp_vector);
    void projectPixels_parallell(cpVector& cp_vector);

    bool resizeLine(Eigen::Vector2f& uv1 ,Eigen::Vector2f& uv2, float& depth1, float& depth2, bool& resized1, bool& resized2);

    inline Camera* clone(){
      return new Camera(*this);
    }

};

typedef std::vector<Camera*> CameraVector;

#endif
