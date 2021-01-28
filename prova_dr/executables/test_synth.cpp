#include "defs.h"
#include "dataset.h"
#include "camera.h"
#include "state.h"
#include "image.h"

using namespace std;
using namespace pr;


int main (int argc, char * argv[]) {


  //############################################################################
  // generate 2 cameras (in this case same orientation, shift on x axis)
  //############################################################################

  CameraVector camera_vector; // initialize vector containing pointers to camera objects for each pose

  int resolution = 640;
  float film = 0.024;
  float lens = 0.035;
  float aspect = 1;

  float offset_x = 0.1;
  float max_depth=2;

  Eigen::Vector3f t_r(0,0,max_depth);
  Eigen::Isometry3f frame_world_wrt_camera_r;
  frame_world_wrt_camera_r.linear().setIdentity();
  frame_world_wrt_camera_r.translation()=t_r;
  Camera* camera_r = new Camera("Camera_r",lens,aspect,film,resolution,frame_world_wrt_camera_r);
  camera_vector.push_back(camera_r);

  Eigen::Vector3f t_m(-offset_x,0,max_depth);
  Eigen::Isometry3f frame_world_wrt_camera_m;
  frame_world_wrt_camera_m.linear().setIdentity();
  frame_world_wrt_camera_m.translation()=t_m;
  Camera* camera_m = new Camera("Camera_m",lens,aspect,film,resolution,frame_world_wrt_camera_m);
  camera_vector.push_back(camera_m);

  camera_r->initImgs();
  camera_m->initImgs();

  //############################################################################
  // generate depth map groundtruth and rgb images of cameras
  //############################################################################


  cpVector cp_vector;
  // generate a "super dense" cloud of points expressed in camera_r frame
  float object_depth=2;
  float left_bound=-object_depth/2;
  float right_bound=object_depth/2+offset_x;
  int density=7000;
  cout << "generating the super dense cloud of points .." << endl;
  for (int x=0; x<density; x++)
    for (int y=0; y<density; y++){
      float x_ = ((float)x/(float)density)*(right_bound-left_bound)+left_bound;
      float y_ = ((float)y/(float)density)*(-left_bound-left_bound)+left_bound;
      float depth = -((sin((x_)*(6*3.14))*sin((x_)*(6*3.14))+sin((y_)*(6*3.14))*sin((y_)*(6*3.14)))/2.0);
      // float depth = -1;

      Cp cp;
      cp.point=Eigen::Vector3f(x_,y_,depth);
      // int clr = depth*(255.0/2.0);
      int clr = 1;
      cp.color=cv::Vec3b(clr,clr,clr);
      cp_vector.push_back(cp);
    }

  camera_r->projectCps(cp_vector);
  camera_m->projectCps(cp_vector);

  camera_r->image_rgb_->show();
  camera_r->depth_map_->show();
  camera_m->image_rgb_->show();
  camera_m->depth_map_->show();
  cv::waitKey(0);

  return 1;
}
