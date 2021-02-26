#include "defs.h"
#include "dataset.h"
#include "camera.h"
#include "image.h"
#include "renderer.h"
#include "environment.h"
#include "utils.h"
#include "dtam_cuda.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#pragma diag_suppress 2739

using namespace std;
using namespace pr;



int main (int argc, char * argv[]) {



  //############################################################################
  // initialization
  //############################################################################

  double t_start=getTime();  // time start for computing computation time
  double t_end=getTime();    // time end for computing computation time

  cpVector cp_vector; // vector of colored points populating the world
  CameraVector camera_vector; // vector containing pointers to camera objects

  EnvGenerator* env_generator = new EnvGenerator(); // environment generator object (pointer)
  Renderer* renderer = new Renderer(cp_vector); // renderer object (pointer)
  // Dtam* dtam = new Dtam(1); // dense mapper and tracker

  //############################################################################
  // generate 2 cameras (in this case same orientation, shift on x axis)
  //############################################################################

  // --------------------------------------
  // generate cameras
  float object_depth=2;

  Camera* camera_r = env_generator->generateCamera("camera_r", 0,0,-object_depth, 0,0,0);
  Camera* camera_m1 = env_generator->generateCamera("camera_m1", 0.1,0.1,-object_depth-0.1, 0,0,0);
  Camera* camera_m2 = env_generator->generateCamera("camera_m2", -0.1,-0.1,-object_depth-0.1, 0,0,0);

  camera_vector.push_back(camera_r);
  camera_vector.push_back(camera_m1);
  camera_vector.push_back(camera_m2);

  // --------------------------------------
  // generate environment
  cerr << "generating environment.." << endl;
  t_start=getTime();

  int density=7000;
  env_generator->generateSinusoidalSurface(object_depth, density, cp_vector);

  t_end=getTime();
  cerr << "environment generation took: " << (t_end-t_start) << " ms" << endl;
  // --------------------------------------


  //############################################################################
  // generate depth map groundtruth and rgb images of cameras
  //############################################################################

  // --------------------------------------
  // rendering environment on cameras
  cerr << "rendering environment on cameras..." << endl;
  t_start=getTime();

  for (Camera* camera : camera_vector)
    renderer->renderImage_naive(cp_vector, camera);

  t_end=getTime();
  cerr << "rendering took: " << (t_end-t_start) << " ms" << endl;
  // --------------------------------------

  Image<float>* depth_map_gt = camera_r->depth_map_->clone("depth map gt");
  // Image<cv::Vec3b>* rgb_image_m_gt = camera_m1->image_rgb_->clone("rgb image m gt");

  // clear depth maps
  for (Camera* camera : camera_vector)
    camera->depth_map_->image_=1.0;

  //
  // //############################################################################
  // // compute depth map
  // //############################################################################
  //
  // // Eigen::Vector2i pixel_coords_r(50,50);
  // Dtam* dtam = new Dtam(camera_vector);
  //
  // cerr << "computing discrete cost volume..." << endl;
  // t_start_projection=getTime();
  //
  // dtam->getDepthMap(100);
  // // dtam->getDepthMap(camera_vector, 100, true);
  // // dtam->getDepthMap(camera_vector, 0.25);
  // // dtam->getDepthMap(camera_vector, 0.25, true);
  //
  // t_end_projection=getTime();
  // cerr << "discrete cost volume computation took: " << (t_end_projection-t_start_projection) << " ms" << endl;
  //
  //
  // for (Camera* camera : camera_vector){
  //   camera->image_rgb_->show(800/resolution);
  //   camera->depth_map_->show(800/resolution);
  // }
  // // depth_map_gt->show(800/resolution);
  // // rgb_image_m_gt->show(800/resolution);
  // cv::waitKey(0);
  //
  //
  //
  // // Eigen::Vector3f o(0,0,0);
  // // camera_r->showWorldFrame(o,0.01,20);
  // // camera_m1->showWorldFrame(o,0.01,20);
  // // camera_r->image_rgb_->show();
  // // camera_m1->image_rgb_->show();
  // // cv::waitKey(0);


  return 1;
}
