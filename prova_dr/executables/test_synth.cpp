#include "defs.h"
#include "dataset.h"
#include "camera.h"
#include "image.h"
#include "utils.h"
#include "dtam.h"
#include <stdio.h>

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

  int resolution = 101;
  float film = 0.024;
  float lens = 0.035;
  float aspect = 1;
  float offset_x = 0.1;
  float offset_y = 0.1;
  float offset_z = -0.1;
  float max_depth=2;

  Eigen::Vector3f t_r(0,0,-object_depth);
  Eigen::Isometry3f frame_world_wrt_camera_r;
  frame_world_wrt_camera_r.linear().setIdentity();
  frame_world_wrt_camera_r.translation()=t_r;
  Eigen::Isometry3f frame_camera_wrt_world_r;
  frame_camera_wrt_world_r = frame_world_wrt_camera_r.inverse();

  Camera* camera_r = new Camera("Camera_r",lens,aspect,film,resolution,max_depth,frame_camera_wrt_world_r,frame_world_wrt_camera_r);
  camera_vector.push_back(camera_r);

  Eigen::Vector3f t_m(-offset_x,-offset_y,-object_depth+offset_z);
  Eigen::Isometry3f frame_world_wrt_camera_m;
  frame_world_wrt_camera_m.linear().setIdentity();
  frame_world_wrt_camera_m.translation()=t_m;
  Eigen::Isometry3f frame_camera_wrt_world_m;
  frame_camera_wrt_world_m = frame_world_wrt_camera_m.inverse();

  Camera* camera_m = new Camera("Camera_m",lens,aspect,film,resolution,max_depth,frame_camera_wrt_world_m,frame_world_wrt_camera_m);
  camera_vector.push_back(camera_m);

  camera_r->initImgs();
  camera_m->initImgs();

  //############################################################################
  // generate depth map groundtruth and rgb images of cameras
  //############################################################################


  cpVector cp_vector;
  // generate a "super dense" cloud of points expressed in camera_r frame
  float left_bound=-object_depth/3;
  float right_bound=(object_depth/3)+(offset_x*object_depth);
  int density=5000;
  cout << "generating the super dense cloud of points .." << endl;
  t_start_projection=getTime();
  for (int x=0; x<density; x++)
    for (int y=0; y<density; y++){
      float x_ = ((float)x/(float)density)*(right_bound-left_bound)+left_bound;
      float y_ = ((float)y/(float)density)*(-left_bound-left_bound)+left_bound;
      float depth = ((sin((x_)*(6*3.14))*sin((x_)*(6*3.14))+sin((y_)*(6*3.14))*sin((y_)*(6*3.14)))/2.0);

      Cp cp;
      cp.point=Eigen::Vector3f(x_,y_,depth);
      int clr_x = ((float)x/(float)density)*255*(sin((x_)*(12*3.14))*sin((x_)*(12*3.14)));
      int clr_y = ((float)y/(float)density)*255*(sin((y_)*(12*3.14))*sin((y_)*(12*3.14)));
      int clr_z = depth*(255.0/max_depth);
      cp.color=cv::Vec3b(clr_x,clr_y,clr_z);
      cp_vector.push_back(cp);
    }
  t_end_projection=getTime();
  cerr << "super dense cloud generation took: " << (t_end_projection-t_start_projection) << " ms" << endl;

  cerr << "projecting super dense cloud of points on cameras..." << endl;
  t_start_projection=getTime();
  // camera_r->projectPixels_parallell(cp_vector);
  // camera_m->projectPixels_parallell(cp_vector);
  t_end_projection=getTime();
  cerr << "projection took: " << (t_end_projection-t_start_projection) << " ms" << endl;


  //############################################################################
  // compute depth map
  //############################################################################

  Eigen::Vector2i pixel_coords_r(50,50);
  Dtam* dtam = new Dtam(1);

  cerr << "computing discrete cost volume..." << endl;
  t_start_projection=getTime();

  dtam->getEpipolarLine(pixel_coords_r, camera_r, camera_m);

  t_end_projection=getTime();

  cerr << "discrete cost volume computation took: " << (t_end_projection-t_start_projection) << " ms" << endl;

  // Eigen::Vector2f uv;
  // Eigen::Vector3f p(-0.67,0.67,0);
  // camera_r->projectPoint( p,uv );

  // Eigen::Vector2i pixel_coords(0,0);
  cv::Vec3b clr(255,0,0);
  camera_r->image_rgb_->setPixel(pixel_coords_r, clr);
  camera_r->image_rgb_->show(800/resolution);

  // cv::waitKey(0);


  //

  // camera_r->image_rgb_->show(800/resolution);
  // camera_r->depth_map_->show(800/resolution);
  // camera_m->image_rgb_->show(800/resolution);
  // camera_m->depth_map_->show(800/resolution);

  // Eigen::Vector3f o(0,0,0);
  // camera_r->showWorldFrame(o,0.01,20);
  // camera_m->showWorldFrame(o,0.01,20);
  // camera_r->image_rgb_->show();
  // camera_m->image_rgb_->show();
  cv::waitKey(0);


  return 1;
}
