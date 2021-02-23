#include "defs.h"
#include "dataset.h"
#include "camera.h"
#include "image.h"
#include "utils.h"
#include "dtam_cuda.cuh"
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;
using namespace pr;


int main (int argc, char * argv[]) {

  double t_start_projection=getTime();
  double t_end_projection=getTime();

  //############################################################################
  // generate 2 cameras (in this case same orientation, shift on x axis)
  //############################################################################

  CameraVector camera_vector; // initialize vector containing pointers to camera objects for each pose

  float object_depth=2;

  int resolution = 200;
  float film = 0.024;
  float lens = 0.035;
  float aspect = 1;
  float offset_x_m1 = -0.1;
  float offset_y_m1 = -0.1;
  float offset_z_m1 = -0.1;
  float offset_x_m2 = 0.1;
  float offset_y_m2 = 0.1;
  float offset_z_m2 = -0.1;
  float max_depth=4.2;

  Eigen::Vector3f t_r(0,0,-object_depth);
  Eigen::Isometry3f frame_world_wrt_camera_r;
  frame_world_wrt_camera_r.linear().setIdentity();
  frame_world_wrt_camera_r.translation()=t_r;
  Eigen::Isometry3f frame_camera_wrt_world_r;
  frame_camera_wrt_world_r = frame_world_wrt_camera_r.inverse();

  Camera* camera_r = new Camera("Camera_r",lens,aspect,film,resolution,max_depth,&frame_camera_wrt_world_r,&frame_world_wrt_camera_r);

  Eigen::Vector3f t_m1(-offset_x_m1,-offset_y_m1,-object_depth+offset_z_m1);
  Eigen::Isometry3f frame_world_wrt_camera_m1;
  frame_world_wrt_camera_m1.linear().setIdentity();
  frame_world_wrt_camera_m1.translation()=t_m1;
  Eigen::Isometry3f frame_camera_wrt_world_m1;
  frame_camera_wrt_world_m1 = frame_world_wrt_camera_m1.inverse();

  Camera* camera_m1 = new Camera("Camera_m1",lens,aspect,film,resolution,max_depth,&frame_camera_wrt_world_m1,&frame_world_wrt_camera_m1);

  Eigen::Vector3f t_m2(-offset_x_m2,-offset_y_m2,-object_depth+offset_z_m2);
  Eigen::Isometry3f frame_world_wrt_camera_m2;
  frame_world_wrt_camera_m2.linear().setIdentity();
  frame_world_wrt_camera_m2.translation()=t_m2;
  Eigen::Isometry3f frame_camera_wrt_world_m2;
  frame_camera_wrt_world_m2 = frame_world_wrt_camera_m2.inverse();

  Camera* camera_m2 = new Camera("Camera_m2",lens,aspect,film,resolution,max_depth,&frame_camera_wrt_world_m2,&frame_world_wrt_camera_m2);


  camera_vector.push_back(camera_r);
  camera_vector.push_back(camera_m1);
  // camera_vector.push_back(camera_m2);



  //############################################################################
  // generate depth map groundtruth and rgb images of cameras
  //############################################################################


  cpVector cp_vector;
  // generate a "super dense" cloud of points expressed in camera_r frame
  float left_bound=-object_depth/3-(0.1*object_depth);
  float right_bound=(object_depth/3)+(0.1*object_depth);
  int density=7000;
  cout << "generating the super dense cloud of points .." << endl;
  t_start_projection=getTime();


  for (int x=0; x<density; x++)
    for (int y=0; y<density; y++){
      float x_ = ((float)x/(float)density)*(right_bound-left_bound)+left_bound;
      float y_ = ((float)y/(float)density)*(-left_bound-left_bound)+left_bound;

      float depth = ((sin((x_)*(6*3.14))*sin((x_)*(6*3.14))+sin((y_)*(6*3.14))*sin((y_)*(6*3.14)))/2.0);

      // int clr_x = ((float)x/(float)density)*255*(sin((x_)*(6*3.14))*sin((x_)*(6*3.14)));
      // int clr_y = ((float)y/(float)density)*255*(sin((y_)*(6*3.14))*sin((y_)*(6*3.14)));
      // int clr_z = depth*(255.0/object_depth);

      int clr_x = ((float)x/(float)density)*255*depth;
      int clr_y = ((float)y/(float)density)*255*depth;
      int clr_z = depth*(255.0/object_depth);


      Cp cp;
      cp.point=Eigen::Vector3f(x_,y_,depth);
      cp.color=cv::Vec3b(clr_x,clr_y,clr_z);
      cp_vector.push_back(cp);
    }
  t_end_projection=getTime();
  cerr << "super dense cloud generation took: " << (t_end_projection-t_start_projection) << " ms" << endl;

  cerr << "projecting super dense cloud of points on cameras..." << endl;
  t_start_projection=getTime();
  for (Camera* camera : camera_vector){
    camera->projectPixels_parallell(cp_vector);
  }
  t_end_projection=getTime();
  cerr << "projection took: " << (t_end_projection-t_start_projection) << " ms" << endl;

  Image<float>* depth_map_gt = camera_r->depth_map_->clone("depth map gt");
  // Image<cv::Vec3b>* rgb_image_m_gt = camera_m1->image_rgb_->clone("rgb image m gt");

  for (Camera* camera : camera_vector)
    camera->depth_map_->image_=1.0;


  //############################################################################
  // compute depth map
  //############################################################################

  // Eigen::Vector2i pixel_coords_r(50,50);
  Dtam* dtam = new Dtam(camera_vector);

  cerr << "computing discrete cost volume..." << endl;
  t_start_projection=getTime();

  dtam->getDepthMap(100);
  // dtam->getDepthMap(camera_vector, 100, true);
  // dtam->getDepthMap(camera_vector, 0.25);
  // dtam->getDepthMap(camera_vector, 0.25, true);



  t_end_projection=getTime();
  cerr << "discrete cost volume computation took: " << (t_end_projection-t_start_projection) << " ms" << endl;


  for (Camera* camera : camera_vector){
    camera->image_rgb_->show(800/resolution);
    camera->depth_map_->show(800/resolution);
  }
  // depth_map_gt->show(800/resolution);
  // rgb_image_m_gt->show(800/resolution);
  cv::waitKey(0);



  // Eigen::Vector3f o(0,0,0);
  // camera_r->showWorldFrame(o,0.01,20);
  // camera_m1->showWorldFrame(o,0.01,20);
  // camera_r->image_rgb_->show();
  // camera_m1->image_rgb_->show();
  // cv::waitKey(0);


  return 1;
}
