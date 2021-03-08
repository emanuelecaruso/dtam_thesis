#include "defs.h"
#include "dataset.h"
#include "camera.h"
#include "image.h"
#include "environment.h"
#include "renderer.h"
#include "utils.h"
#include "dtam.h"
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

  int resolution = 600;

  Environment* environment = new Environment(resolution); // environment generator object (pointer)
  Renderer* renderer = new Renderer(); // renderer object (pointer)
  Dtam* dtam = new Dtam(); // dense mapper and tracker

  //############################################################################
  // generate 2 cameras (in this case same orientation, shift on x axis)
  //############################################################################

  // --------------------------------------
  // generate cameras
  float object_depth=2;

  environment->generateCamera("camera_r", 0,0,-object_depth, 0,0,0);
  environment->generateCamera("camera_m1", 0.1,0.1,-object_depth-0.1, 0,0,0);
  // environment->generateCamera("camera_m2", -0.1,-0.1,-object_depth-0.1, 0,0,0);


  // --------------------------------------
  // generate environment
  cerr << "generating environment.." << endl;
  t_start=getTime();

  int density=6000;
  environment->generateSinusoidalSurface(object_depth, density);

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

  renderer->renderImage_parallel_cpu(environment);

  t_end=getTime();
  cerr << "rendering took: " << (t_end-t_start) << " ms" << endl;
  // --------------------------------------

  Camera* camera_r = environment->camera_vector_[0];
  Image<float>* depth_map_gt = camera_r->depth_map_->clone("depth map gt");
  // Image<cv::Vec3b>* rgb_image_m_gt = camera_m1->image_rgb_->clone("rgb image m gt");

  // clear depth maps
  for (Camera* camera : environment->camera_vector_)
    camera->depth_map_->image_=1.0;


  //############################################################################
  // compute depth map
  //############################################################################

  // --------------------------------------
  // load cameras for dtam

  dtam->loadCameras(environment->camera_vector_);
  dtam->setReferenceCamera(0);

  // --------------------------------------
  cerr << "computing discrete cost volume..." << endl;
  t_start=getTime();

  dtam->getDepthMap(64);
  // dtam->getDepthMap(64, true);

  //
  t_end=getTime();
  cerr << "discrete cost volume computation took: " << (t_end-t_start) << " ms" << endl;

  // --------------------------------------
  // show camera rgb images and depth maps
  for (Camera* camera : environment->camera_vector_){
    camera->image_rgb_->show(800/camera->resolution_);
    camera->depth_map_->show(800/camera->resolution_);
  }
  depth_map_gt->show(800/environment->camera_vector_[0]->resolution_);
  // rgb_image_m_gt->show(800/resolution);
  cv::waitKey(0);
  // --------------------------------------

  return 1;
}
