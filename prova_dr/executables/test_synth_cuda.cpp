#include "defs.h"
#include "dataset.h"
#include "camera_cpu.cuh"
#include "image.h"
#include "environment.cuh"
#include "renderer.cuh"
#include "utils.h"
#include "dtam_cuda.cuh"
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;
using namespace pr;



int main (int argc, char * argv[]) {



  //############################################################################
  // initialization
  //############################################################################

  double t_start=getTime();  // time start for computing computation time
  double t_end=getTime();    // time end for computing computation time

  Environment* environment = new Environment(); // environment generator object (pointer)
  Renderer* renderer = new Renderer(); // renderer object (pointer)
  Dtam* dtam = new Dtam(); // dense mapper and tracker

  //############################################################################
  // generate cameras (in this case same orientation)
  //############################################################################

  // --------------------------------------
  // generate cameras
  float object_depth=2;

  environment->generateCamera("camera_r", 0,0,-object_depth, 0,0,0);
  environment->generateCamera("camera_m1", 0.1,0.1,-object_depth-0.1, 0,0,0);
  environment->generateCamera("camera_m2", -0.1,-0.1,-object_depth-0.1, 0,0,0);

  // --------------------------------------
  // generate environment
  cout << "generating environment.." << endl;
  t_start=getTime();

  int density=6000; // NOTE: to work cp_vectorwith cuda, density^2 must be a multiplier of 32
  environment->generateSinusoidalSurface(object_depth, density);

  t_end=getTime();
  cout << "environment generation took: " << (t_end-t_start) << " ms" << endl;
  // --------------------------------------


  //############################################################################
  // generate depth map groundtruth and rgb images of cameras
  //############################################################################

  // --------------------------------------
  // rendering environment on cameras
  cout << "rendering environment on cameras..." << endl;
  t_start=getTime();

  renderer->renderImages_parallel_gpu(environment);

  t_end=getTime();
  cout << "rendering took: " << (t_end-t_start) << " ms" << endl;
  // --------------------------------------

  Camera_cpu* camera_r = environment->camera_vector_cpu_[0];
  Image<float>* depth_map_gt = camera_r->depth_map_->clone("depth map gt");
  // Image<cv::Vec3b>* rgb_image_m_gt = camera_m1->image_rgb_->clone("rgb image m gt");

  // clear depth maps
  for (Camera_cpu* camera : environment->camera_vector_cpu_)
    camera->depth_map_->image_=1.0;



  //############################################################################
  // compute depth map
  //############################################################################

  cout << "computing discrete cost volume..." << endl;
  t_start=getTime();

  dtam->getDepthMap(100, environment->camera_vector_cpu_, environment->camera_vector_gpu_);
  // dtam->getDepthMap(camera_vector, 100, true);
  // dtam->getDepthMap(camera_vector, 0.25);
  // dtam->getDepthMap(camera_vector, 0.25, true);

  t_end=getTime();
  cout << "discrete cost volume computation took: " << (t_end-t_start) << " ms" << endl;


  for (Camera_cpu* camera : environment->camera_vector_cpu_){
    camera->image_rgb_->show(800/camera->resolution_);
    camera->depth_map_->show(800/camera->resolution_);
  }
  depth_map_gt->show(800/camera_r->resolution_);
  // rgb_image_m_gt->show(800/resolution);
  cv::waitKey(0);
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
