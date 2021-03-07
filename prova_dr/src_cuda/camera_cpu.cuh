#pragma once
#include "defs.h"
#include "image.h"
#include "camera_gpu.cuh"


using namespace pr;


class Camera_cpu{
  public:

    std::string name_;
    float lens_;
    float aspect_;
    float width_;
    int resolution_;
    float max_depth_;
    Eigen::Matrix3f K_;
    Eigen::Matrix3f Kinv_;
    Image<float>* depth_map_;
    Image<cv::Vec3b>* image_rgb_;
    Eigen::Isometry3f* frame_world_wrt_camera_;
    Eigen::Isometry3f* frame_camera_wrt_world_;
    cv::cuda::GpuMat depth_map_gpu_;
    cv::cuda::GpuMat image_rgb_gpu_;
    // camera data for dtam
    Eigen::Matrix3f T_r;
    Eigen::Vector3f T_t;
    Eigen::Vector2f cam_r_projected_on_cam_m;
    float cam_r_depth_on_camera_m;
    bool cam_r_in_front;
    
    Camera_cpu(std::string name, float lens, float aspect, float width, int resolution,
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

       // compute camera matrix and its inverse
       K_ <<
         lens_,   0   ,     -width_/2       ,
         0    ,  -lens_, -(width_/aspect_)/2,
         0    ,   0   ,       -1            ;
       Kinv_=K_.inverse();
    };

    // bool extractCameraMatrix(Eigen::Matrix3f& K);
    // void showWorldFrame(Eigen::Vector3f origin, float delta,int length);

    void clearImgs();

    void gpuFree();
    Camera_gpu* getCamera_gpu();

    void printMembers();
    void computeDataForDtam(int index_r);

    void pixelCoords2uv(Eigen::Vector2i& pixel_coords, Eigen::Vector2f& uv);
    void uv2pixelCoords( Eigen::Vector2f& uv, Eigen::Vector2i& pixel_coords);

    void pointAtDepth(Eigen::Vector2f& uv, float depth, Eigen::Vector3f& p);
    bool projectPoint(Eigen::Vector3f& p, Eigen::Vector2f& uv, float& p_cam_z );
    // bool projectPixel(Cp& p);
    // void projectPixels(cpVector& cp_vector);
    // void projectPixels_parallell(cpVector& cp_vector);

    bool resizeLine(Eigen::Vector2f& uv1 ,Eigen::Vector2f& uv2, float& depth1, float& depth2, bool& resized1, bool& resized2);

    inline Camera_cpu* clone(){
      return new Camera_cpu(*this);
    }

};

typedef std::vector<Camera_cpu*> CameraVector_cpu;
