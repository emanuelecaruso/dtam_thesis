#include "dtam.h"
#include <math.h>
#include "utils.h"

// bool Dtam::get1stDepthWithUV(Camera* camera_r, Camera* camera_m, Eigen::Vector2f& uv_r, Eigen::Vector2f& uv_m, float& depth){
//
//   Eigen::Isometry3f T = (*(camera_m->frame_world_wrt_camera_))*(*(camera_r->frame_camera_wrt_world_));
//   auto r=T.linear();
//   auto t=T.translation();
//   float f = camera_r->lens_;
//   float w=camera_m->width_;
//   float h=camera_m->width_/camera_m->aspect_;
//   depth = (2*f*(f+t(2)))/(2*f*r(2,2)-2*r(2,0)*uv_r.x()+r(2,0)*w-r(2,1)*(h-2*uv_r.y()));
//   uv_m.x() = t(0)+(w/2)-depth*r(0,2)+((depth*r(0,0)*(2*uv_r.x()-w))/(2*f))+((depth*r(0,1)*(h-2*uv_r.y()))/(2*f));
//   uv_m.y() = (h/2)-t(1)+depth*r(1,2)-((depth*r(1,0)*(2*uv_r.x()-w))/(2*f))-((depth*r(1,1)*(h-2*uv_r.y()))/(2*f));
// }
//
// struct cameraData{
//   Eigen::Vector2f uv1, uv2, uv1_fixed, uv2_fixed;
//   float depth1_m, depth2_m, depth1_m_fixed, depth2_m_fixed;
//   Eigen::Matrix3f r;
//   Eigen::Vector3f t;
//   Eigen::Vector2f cam_r_projected_on_cam_m;
//   float cam_r_depth_on_camera_m;
//   bool cam_r_in_front;
// };

void Dtam::loadCameras(CameraVector camera_vector){
  camera_vector_= camera_vector;
}

void Dtam::addCamera(Camera* camera){
  camera_vector_.push_back(camera);
}

bool Dtam::setReferenceCamera(int index_r){

  int num_cameras = camera_vector_.size();

  if (index_r<0 || index_r>=num_cameras)
    return false;

  index_r_ = index_r;

  Camera* camera_r = camera_vector_[index_r];
  Eigen::Vector3f camera_r_p = camera_r->frame_camera_wrt_world_->translation();

  for (int camera_iterator=0; camera_iterator<num_cameras; camera_iterator++){

    if (camera_iterator!=index_r){
      Camera* camera_m = camera_vector_[camera_iterator];

      // project camera_r on camera_m
      Eigen::Vector2f cam_r_projected_on_cam_m;
      float cam_r_depth_on_camera_m;
      bool cam_r_in_front = camera_m->projectPoint(camera_r_p, cam_r_projected_on_cam_m, cam_r_depth_on_camera_m);

      Eigen::Isometry3f T = (*(camera_m->frame_world_wrt_camera_))*(*(camera_r->frame_camera_wrt_world_));
      Eigen::Matrix3f r=T.linear();
      Eigen::Vector3f t=T.translation();

      camera_m->T_r=r;
      camera_m->T_t=t;
      camera_m->cam_r_projected_on_cam_m=cam_r_projected_on_cam_m;
      camera_m->cam_r_depth_on_camera_m=cam_r_depth_on_camera_m;
      camera_m->cam_r_in_front=cam_r_in_front;

    }

  }
  return true;

}


void Dtam::getDepthMap(int num_interpolations, bool check){
  double t_start;
  double t_end;


  // reference camera
  Camera* camera_r = camera_vector_[index_r_];
  float depth1_r=camera_r->lens_;
  float depth2_r=camera_r->max_depth_;

  int cols = camera_r->depth_map_->image_.cols;
  int rows = camera_r->depth_map_->image_.rows;

  float f = camera_r->lens_;
  float w=camera_r->width_;
  float h=camera_r->width_/camera_r->aspect_;

  cost_matrix_->initImage(rows,cols*num_interpolations);
  cost_matrix_->setAllPixels(999999);
  n_valid_proj_matrix_->initImage(rows,cols*num_interpolations);
  n_valid_proj_matrix_->setAllPixels(0);



  for (int row = 0; row<rows; row++){
    for (int col = 0; col<cols; col++){

      if (check){row= rows*0.36; col=cols*0.45;}

      Eigen::Vector2i pixel_coords_r(col,row);
      cv::Vec3b clr_r;
      camera_r->image_rgb_->evalPixel(pixel_coords_r,clr_r);

      // query point
      Eigen::Vector3f query_p;
      Eigen::Vector2f uv_r;
      camera_r->pixelCoords2uv(pixel_coords_r, uv_r);
      camera_r->pointAtDepth(uv_r, depth2_r, query_p);

      bool invalid_pxl=false;

      int cost_min = 999999;
      float depth_min = -1;
      // int iterator = 0;

      for (int i=0; i<num_interpolations; i++){
      // for (int i=0; i<5; i++){

        float ratio_depth_r = (float)i/((float)num_interpolations-1);
        float depth_r = depth1_r+ratio_depth_r*(depth2_r-depth1_r);

        int cost_i = -1;
        int num_valid_projections = 0;

        for (int camera_iterator=0; camera_iterator<camera_vector_.size(); camera_iterator++){
          Camera* camera_m = camera_vector_[camera_iterator];

          if (camera_iterator!=index_r_){
            auto cam_r_projected_on_cam_m = camera_vector_[camera_iterator]->cam_r_projected_on_cam_m;
            auto cam_r_depth_on_camera_m = camera_vector_[camera_iterator]->cam_r_depth_on_camera_m;
            auto r=camera_vector_[camera_iterator]->T_r;
            auto t=camera_vector_[camera_iterator]->T_t;
            auto cam_r_in_front = camera_vector_[camera_iterator]->cam_r_in_front;

            // project query point
            Eigen::Vector2f query_p_projected_on_cam_m;
            float query_depth_on_camera_m;
            bool query_in_front = camera_m->projectPoint(query_p, query_p_projected_on_cam_m, query_depth_on_camera_m);

            // initializations
            Eigen::Vector2f uv1_fixed, uv2_fixed;
            float depth1_m_fixed, depth2_m_fixed;

            if (check){ cv::Vec3b clr = cv::Vec3b(0,0,255);
              camera_r->image_rgb_->setPixel(pixel_coords_r,clr); }

            // if both camera r and query point are on back of camera m return false
            if (!query_in_front && !cam_r_in_front)
              continue;
            // if query point is in front of camera m whereas camera r is on the back
            else if (query_in_front && !cam_r_in_front){
              uv1_fixed=cam_r_projected_on_cam_m;
              uv2_fixed=query_p_projected_on_cam_m;
              depth1_m_fixed=cam_r_depth_on_camera_m;
              depth2_m_fixed=query_depth_on_camera_m;
            }
            // if camera r is in front of camera m whereas query point is on the back
            else if (!query_in_front && cam_r_in_front){
              // TODO
              continue;
            }
            // if both camera r and query point are in front of camera m
            else {

              uv1_fixed=cam_r_projected_on_cam_m;
              uv2_fixed=query_p_projected_on_cam_m;
              depth1_m_fixed=cam_r_depth_on_camera_m;
              depth2_m_fixed=query_depth_on_camera_m;

            }

            float depth_m = depth_r*r(2,2)-t(2)-((depth_r*r(2,0)*(2*uv_r.x()-w))/(2*f))-((depth_r*r(2,1)*(-2*uv_r.y()+h))/(2*f));
            float ratio_invdepth_m = ((1.0/depth_m)-(1.0/depth1_m_fixed))/((1.0/depth2_m_fixed)-(1.0/depth1_m_fixed));

            Eigen::Vector2f uv_current;
            uv_current.x()=uv1_fixed.x()+ratio_invdepth_m*(uv2_fixed.x()-uv1_fixed.x()) ;
            uv_current.y()=uv1_fixed.y()+ratio_invdepth_m*(uv2_fixed.y()-uv1_fixed.y()) ;

            Eigen::Vector2i pixel_current;
            camera_m->uv2pixelCoords( uv_current, pixel_current);
            cv::Vec3b clr_current;
            bool flag = camera_m->image_rgb_->evalPixel(pixel_current,clr_current);
            if (!flag)
              continue;

            num_valid_projections++;

            int cost_current = mseBetween2Colors(clr_r, clr_current);


            if (cost_i<0)
              cost_i=0;

            cost_i+=cost_current;

            if (check){
              // std::cout << "cam_r_projected_on_cam_m " << cam_r_projected_on_cam_m << std::endl;
              cv::Vec3b clr = cv::Vec3b(ratio_depth_r*255,0,ratio_depth_r*255);
              cv::Vec3b red = cv::Vec3b(0,0,255);
              cv::Vec3b blue = cv::Vec3b(255,0,0);
              if (i==0)
                camera_m->image_rgb_->setPixel(pixel_current,red);
              else if(i==num_interpolations-1)
                camera_m->image_rgb_->setPixel(pixel_current,blue);
              else
                camera_m->image_rgb_->setPixel(pixel_current,clr);
            }

          }

        }

        if (cost_i>0){

          cost_matrix_->setPixel(row,col+cols*i,(cost_i/num_valid_projections));
          n_valid_proj_matrix_->setPixel(row,col+cols*i,num_valid_projections);

          if (cost_i<cost_min){
            depth_min=depth_r;
            cost_min=cost_i/num_valid_projections;
          }
        }

      }

      if (depth_min>=0){
        float depth_value = depth_min/camera_r->max_depth_;
        camera_r->depth_map_->setPixel(pixel_coords_r,depth_value);
      }


      if (check)  {break;}
    }
    if (check)  {break;}
  }


}
