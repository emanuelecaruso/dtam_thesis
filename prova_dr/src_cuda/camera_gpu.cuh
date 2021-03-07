#pragma once
#include "defs.h"
#include "image.h"
#include <cuda_runtime.h>

using namespace pr;

class Camera_gpu{
  public:

    std::string name_;
    float lens_;
    float aspect_;
    float width_;
    int resolution_;
    float max_depth_;
    Eigen::Matrix3f K_;
    Eigen::Matrix3f Kinv_;
    cv::cuda::PtrStepSz<float> depth_map_;
    cv::cuda::PtrStepSz<uchar3> image_rgb_;
    Eigen::Isometry3f frame_world_wrt_camera_;
    Eigen::Isometry3f frame_camera_wrt_world_;
    // camera data for dtam
    Eigen::Matrix3f T_r;
    Eigen::Vector3f T_t;
    Eigen::Vector2f cam_r_projected_on_cam_m;
    float cam_r_depth_on_camera_m;
    bool cam_r_in_front;
    Eigen::Vector2f  uv1_fixed;

    Camera_gpu(std::string name, float lens, float aspect, float width, int resolution,
       float max_depth, Eigen::Matrix3f K, Eigen::Matrix3f Kinv, Eigen::Isometry3f frame_camera_wrt_world,
       Eigen::Isometry3f frame_world_wrt_camera, cv::cuda::PtrStepSz<float> depth_map, cv::cuda::PtrStepSz<uchar3> image_rgb){
       name_ = name;
       lens_ = lens;
       aspect_ = aspect;
       width_ = width;
       resolution_ = resolution;
       max_depth_ = max_depth;
       K_ = K;
       Kinv_ = Kinv;
       frame_camera_wrt_world_ = frame_camera_wrt_world;
       frame_world_wrt_camera_ = frame_world_wrt_camera;
       depth_map_ = depth_map;
       image_rgb_ = image_rgb;

    };

    // __device__ void clearImgs();

    void printMembers();

    __device__ void pixelCoords2uv(Eigen::Vector2i& pixel_coords, Eigen::Vector2f& uv);
    __device__ void uv2pixelCoords( Eigen::Vector2f& uv, Eigen::Vector2i& pixel_coords);

    __device__ void pointAtDepth(Eigen::Vector2f& uv, float depth, Eigen::Vector3f& p);
    __device__ bool projectPoint(Eigen::Vector3f& p, Eigen::Vector2f& uv, float& p_cam_z );
    // __device__ bool projectPixel(Cp& p);
    // __device__ void projectPixels(cpVector& cp_vector);


    inline Camera_gpu* clone(){
      return new Camera_gpu(*this);
    }

};

typedef std::vector<Camera_gpu*> CameraVector_gpu;
