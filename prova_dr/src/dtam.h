// #pragma once
// #include "camera.h"
// #include "environment.h"
// #include "image.h"
//
// using namespace pr;
//
// class Dtam{
//
//   struct cameraDataForDtam{
//     Eigen::Matrix3f T_r;
//     Eigen::Vector3f T_t;
//     Eigen::Vector2f cam_r_projected_on_cam_m;
//     float cam_r_depth_on_camera_m;
//     bool cam_r_in_front;
//   };
//
//   public:
//     CameraVector camera_vector_;
//     int index_r_;
//     int num_interpolations_;
//     float* depth_r_array_;
//     Image<int>* cost_matrix_;
//     Image<int>* n_valid_proj_matrix_;
//     cameraDataForDtam camera_data_for_dtam_;
//
//     Dtam(Environment* environment, int num_interpolations){
//       num_interpolations_=num_interpolations;
//
//       cost_matrix_ = new Image<int>("cost matrix");
//       n_valid_proj_matrix_ = new Image<int>("matrix of valid projections");
//
//       int rows = environment->resolution_/environment->aspect_;
//       int cols = environment->resolution_;
//       float depth1_r=environment->lens_;
//       float depth2_r=environment->max_depth_;
//       depth_r_array_ = new float[num_interpolations_];
//
//       for (int i=0; i<num_interpolations_; i++){
//         float ratio_depth_r = (float)i/((float)num_interpolations_-1);
//         // float depth_r = 1.0/((1.0/depth1_r)+ratio_depth_r*((1.0/depth2_r)-(1.0/depth1_r)));
//         float depth_r = depth1_r+ratio_depth_r*(depth2_r-depth1_r);
//         depth_r_array_[i]=depth_r;
//       }
//
//     }
//
//     void loadCameras(CameraVector camera_vector);
//     void addCamera(Camera* camera);
//     bool setReferenceCamera(int index_r);
//     void prepareCameraForDtam(int index_m);
//
//     void updateDepthMap( int index_m, bool check=false);
//     void updateDepthMap_parallel_cpu( int index_m, bool check=false);
// };
