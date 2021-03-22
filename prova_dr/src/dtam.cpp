#include "dtam.h"
#include <math.h>
#include "utils.h"


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

  int cols = camera_vector_[index_r_]->depth_map_->image_.cols;
  int rows = camera_vector_[index_r_]->depth_map_->image_.rows;

  cost_matrix_->initImage(rows,cols*num_interpolations_);
  cost_matrix_->setAllPixels(999999);
  n_valid_proj_matrix_->initImage(rows,cols*num_interpolations_);
  n_valid_proj_matrix_->setAllPixels(0);

  return true;

}

void Dtam::prepareCameraForDtam(int index_m){
  Camera* camera_r = camera_vector_[index_r_];
  Eigen::Vector3f camera_r_p = camera_r->frame_camera_wrt_world_->translation();
  Camera* camera_m = camera_vector_[index_m];

  // project camera_r on camera_m
  Eigen::Vector2f cam_r_projected_on_cam_m;
  float cam_r_depth_on_camera_m;
  bool cam_r_in_front = camera_m->projectPoint(camera_r_p, cam_r_projected_on_cam_m, cam_r_depth_on_camera_m);

  Eigen::Isometry3f T = (*(camera_m->frame_world_wrt_camera_))*(*(camera_r->frame_camera_wrt_world_));
  Eigen::Matrix3f r=T.linear();
  Eigen::Vector3f t=T.translation();

  camera_data_for_dtam_.T_r=r;
  camera_data_for_dtam_.T_t=t;
  camera_data_for_dtam_.cam_r_projected_on_cam_m=cam_r_projected_on_cam_m;
  camera_data_for_dtam_.cam_r_depth_on_camera_m=cam_r_depth_on_camera_m;
  camera_data_for_dtam_.cam_r_in_front=cam_r_in_front;
}



void Dtam::updateDepthMap(int index_m, bool check){
  double t_start;
  double t_end;

  // reference camera
  Camera* camera_r = camera_vector_[index_r_];
  Camera* camera_m = camera_vector_[index_m];

  int cols = camera_r->depth_map_->image_.cols;
  int rows = camera_r->depth_map_->image_.rows;

  float depth1_r=camera_r->lens_;
  float depth2_r=camera_r->max_depth_;

  float f = camera_r->lens_;
  float w=camera_r->width_;
  float h=camera_r->width_/camera_r->aspect_;

  auto cam_r_projected_on_cam_m =camera_data_for_dtam_.cam_r_projected_on_cam_m;
  auto cam_r_depth_on_camera_m =camera_data_for_dtam_.cam_r_depth_on_camera_m;
  auto r=camera_data_for_dtam_.T_r;
  auto t=camera_data_for_dtam_.T_t;
  auto cam_r_in_front =camera_data_for_dtam_.cam_r_in_front;

  for (int j = 0; j<rows*cols; j++){

    int row = j/cols;
    int col = j%cols;

    if (check){row= rows*0.97; col=cols*0.03;}

    Eigen::Vector2i pixel_coords_r(col,row);
    cv::Vec3b clr_r;
    camera_r->image_rgb_->evalPixel(pixel_coords_r,clr_r);

    // query point
    Eigen::Vector3f query_p;
    Eigen::Vector2f uv_r;
    camera_r->pixelCoords2uv(pixel_coords_r, uv_r);
    camera_r->pointAtDepth(uv_r, depth2_r, query_p);
    Eigen::Vector2f query_p_projected_on_cam_m;
    float query_depth_on_camera_m;
    bool query_in_front = camera_m->projectPoint(query_p, query_p_projected_on_cam_m, query_depth_on_camera_m);

    // if both camera r and query point are on back of camera m return false or
    // if camera r is in front of camera m whereas query point is on the back
    if ((!query_in_front && !cam_r_in_front) || (!query_in_front && cam_r_in_front))
      continue;

    int cost_min = 999999;
    float depth_min = -1;

    // initializations
    Eigen::Vector2f uv1_fixed, uv2_fixed;
    float depth1_m_fixed, depth2_m_fixed;

    uv1_fixed=cam_r_projected_on_cam_m;
    uv2_fixed=query_p_projected_on_cam_m;
    depth1_m_fixed=cam_r_depth_on_camera_m;
    depth2_m_fixed=query_depth_on_camera_m;

    Eigen::Vector2i pixel_current;


    for (int i=0; i<num_interpolations_; i++){

      float depth_r = depth_r_array_[i];
      float depth_m = depth_r*r(2,2)-t(2)-((depth_r*r(2,0)*(2*uv_r.x()-w))/(2*f))-((depth_r*r(2,1)*(-2*uv_r.y()+h))/(2*f));
      float ratio_invdepth_m = ((1.0/depth_m)-(1.0/depth1_m_fixed))/((1.0/depth2_m_fixed)-(1.0/depth1_m_fixed));

      Eigen::Vector2f uv_current;
      uv_current.x()=uv1_fixed.x()+ratio_invdepth_m*(uv2_fixed.x()-uv1_fixed.x()) ;
      uv_current.y()=uv1_fixed.y()+ratio_invdepth_m*(uv2_fixed.y()-uv1_fixed.y()) ;


      camera_m->uv2pixelCoords( uv_current, pixel_current);
      cv::Vec3b clr_current;
      bool flag = camera_m->image_rgb_->evalPixel(pixel_current,clr_current);

      int cost_i;
      int col_ = col+cols*i;
      int num_valid_projections;

      if (!flag){
        cost_matrix_->evalPixel(row,col_,cost_i);
      }
      else{

        cost_matrix_->evalPixel(row,col_,cost_i);
        n_valid_proj_matrix_->evalPixel(row,col_,num_valid_projections);

        int cost_current = mseBetween2Colors(clr_r, clr_current);
        if (cost_i==999999)
          cost_i=cost_current;
        else{
          cost_i=(cost_i*num_valid_projections+cost_current)/(num_valid_projections+1);
          // cost_i=cost_current;
        }
        if (check){
          std::cout  << "i " << i<< ", cost_i " << cost_i << std::endl;
          cv::Vec3b magenta = cv::Vec3b(255,0,255);
          cv::Vec3b red = cv::Vec3b(0,0,255);
          cv::Vec3b blue = cv::Vec3b(255,0,0);
          if (i==0)
            camera_m->image_rgb_->setPixel(pixel_current,red);
          else if(i==num_interpolations_-1)
            camera_m->image_rgb_->setPixel(pixel_current,blue);
          else
            camera_m->image_rgb_->setPixel(pixel_current,magenta);
          camera_r->image_rgb_->setPixel(pixel_coords_r,red);
        }
      }


      if (cost_i<999999){

        cost_matrix_->setPixel(row,col_,cost_i);
        n_valid_proj_matrix_->setPixel(row,col_,(num_valid_projections+1));

        if (cost_i<cost_min){
          depth_min=depth_r;
          cost_min=cost_i;
        }
      }

    }


    if (depth_min>=0){
      float depth_value = depth_min/camera_r->max_depth_;
      camera_r->depth_map_->setPixel(pixel_coords_r,depth_value);
    }

    if (check)  {break;}
  }

}

void Dtam::updateDepthMap_parallel_cpu(int index_m, bool check){
  double t_start;
  double t_end;

  // reference camera
  Camera* camera_r = camera_vector_[index_r_];
  Camera* camera_m = camera_vector_[index_m];

  int cols = camera_r->depth_map_->image_.cols;
  int rows = camera_r->depth_map_->image_.rows;

  float depth1_r=camera_r->lens_;
  float depth2_r=camera_r->max_depth_;

  float f = camera_r->lens_;
  float w=camera_r->width_;
  float h=camera_r->width_/camera_r->aspect_;

  auto cam_r_projected_on_cam_m =camera_data_for_dtam_.cam_r_projected_on_cam_m;
  auto cam_r_depth_on_camera_m =camera_data_for_dtam_.cam_r_depth_on_camera_m;
  auto T_r=camera_data_for_dtam_.T_r;
  auto T_t=camera_data_for_dtam_.T_t;
  auto cam_r_in_front =camera_data_for_dtam_.cam_r_in_front;

  // camera->clearImgs();
  const size_t nloop = rows*cols;
  const size_t nthreads = std::thread::hardware_concurrency();
  {
    // Pre loop
    std::vector<std::thread> threads(nthreads);
    std::mutex critical;
    for(int t = 0;t<nthreads;t++)
    {
      threads[t] = std::thread(std::bind(
        [&](const int bi, const int ei, const int t)
        {
          // loop over all items
          for(int j = bi;j<ei;j++)
          {
            // inner loop
            {
                int row = j/cols;
                int col = j%cols;

                if (check){row= rows*0.97; col=cols*0.03;}

                Eigen::Vector2i pixel_coords_r(col,row);
                cv::Vec3b clr_r;
                camera_r->image_rgb_->evalPixel(pixel_coords_r,clr_r);

                // query point
                Eigen::Vector3f query_p;
                Eigen::Vector2f uv_r;
                camera_r->pixelCoords2uv(pixel_coords_r, uv_r);
                camera_r->pointAtDepth(uv_r, depth2_r, query_p);
                Eigen::Vector2f query_p_projected_on_cam_m;
                float query_depth_on_camera_m;
                bool query_in_front = camera_m->projectPoint(query_p, query_p_projected_on_cam_m, query_depth_on_camera_m);

                // if both camera r and query point are on back of camera m return false or
                // if camera r is in front of camera m whereas query point is on the back
                if ((!query_in_front && !cam_r_in_front) || (!query_in_front && cam_r_in_front))
                  continue;

                int cost_min = 999999;
                float depth_min = -1;

                // initializations
                Eigen::Vector2f uv1_fixed, uv2_fixed;
                float depth1_m_fixed, depth2_m_fixed;

                uv1_fixed=cam_r_projected_on_cam_m;
                uv2_fixed=query_p_projected_on_cam_m;
                depth1_m_fixed=cam_r_depth_on_camera_m;
                depth2_m_fixed=query_depth_on_camera_m;

                Eigen::Vector2i pixel_current;


                for (int i=0; i<num_interpolations_; i++){

                  float depth_r = depth_r_array_[i];
                  float depth_m = depth_r*T_r(2,2)-T_t(2)-((depth_r*T_r(2,0)*(2*uv_r.x()-w))/(2*f))-((depth_r*T_r(2,1)*(-2*uv_r.y()+h))/(2*f));
                  float ratio_invdepth_m = ((1.0/depth_m)-(1.0/depth1_m_fixed))/((1.0/depth2_m_fixed)-(1.0/depth1_m_fixed));

                  Eigen::Vector2f uv_current;
                  uv_current.x()=uv1_fixed.x()+ratio_invdepth_m*(uv2_fixed.x()-uv1_fixed.x()) ;
                  uv_current.y()=uv1_fixed.y()+ratio_invdepth_m*(uv2_fixed.y()-uv1_fixed.y()) ;


                  camera_m->uv2pixelCoords( uv_current, pixel_current);
                  cv::Vec3b clr_current;
                  bool flag = camera_m->image_rgb_->evalPixel(pixel_current,clr_current);

                  int cost_i;
                  int col_ = col+cols*i;
                  int num_valid_projections;

                  if (!flag){
                    cost_matrix_->evalPixel(row,col_,cost_i);
                  }
                  else{

                    cost_matrix_->evalPixel(row,col_,cost_i);
                    n_valid_proj_matrix_->evalPixel(row,col_,num_valid_projections);

                    int cost_current = mseBetween2Colors(clr_r, clr_current);
                    if (cost_i==999999)
                      cost_i=cost_current;
                    else{
                      cost_i=(cost_i*num_valid_projections+cost_current)/(num_valid_projections+1);
                      // cost_i=cost_current;
                    }
                    if (check){
                      std::cout  << "i " << i<< ", cost_i " << cost_i << std::endl;
                      cv::Vec3b magenta = cv::Vec3b(255,0,255);
                      cv::Vec3b red = cv::Vec3b(0,0,255);
                      cv::Vec3b blue = cv::Vec3b(255,0,0);
                      if (i==0)
                        camera_m->image_rgb_->setPixel(pixel_current,red);
                      else if(i==num_interpolations_-1)
                        camera_m->image_rgb_->setPixel(pixel_current,blue);
                      else
                        camera_m->image_rgb_->setPixel(pixel_current,magenta);
                      camera_r->image_rgb_->setPixel(pixel_coords_r,red);
                    }
                  }


                  if (cost_i<999999){

                    cost_matrix_->setPixel(row,col_,cost_i);
                    n_valid_proj_matrix_->setPixel(row,col_,(num_valid_projections+1));

                    if (cost_i<cost_min){
                      depth_min=depth_r;
                      cost_min=cost_i;
                    }
                  }

                }


                if (depth_min>=0){
                  float depth_value = depth_min/camera_r->max_depth_;
                  camera_r->depth_map_->setPixel(pixel_coords_r,depth_value);
                }

                if (check)  {break;}

            }
          }
        },t*nloop/nthreads,(t+1)==nthreads?nloop:(t+1)*nloop/nthreads,t));
    }
    std::for_each(threads.begin(),threads.end(),[](std::thread& x){x.join();});
    // Post loop
  }



}
