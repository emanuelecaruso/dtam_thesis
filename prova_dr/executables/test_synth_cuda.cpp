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
  // choose parameters
  //############################################################################

  int resolution = 600;
  float aspect = 1;
  int num_interpolations = 64;

  //############################################################################
  // initialization
  //############################################################################

  double t_start=getTime();  // time start for computing computation time
  double t_end=getTime();    // time end for computing computation time

  Environment* environment = new Environment(resolution, aspect); // environment generator object (pointer)
  Renderer* renderer = new Renderer(); // renderer object (pointer)
  Dtam* dtam = new Dtam(environment, num_interpolations); // dense mapper and tracker

  //############################################################################
  // generate cameras (in this case same orientation)
  //############################################################################

  // --------------------------------------
  // generate cameras
  float object_depth=2;
  float offset=-0.05;
  float offset_depth=-0.05;

  environment->generateCamera("camera_r", 0,0,-object_depth, 0,0,0);
  environment->generateCamera("camera_m1", offset,offset,-object_depth-offset_depth, 0,0,0);
  environment->generateCamera("camera_m2", -offset,offset,-object_depth-offset_depth, 0,0,0);
  environment->generateCamera("camera_m3", 0,offset,-object_depth-offset_depth, 0,0,0);
  environment->generateCamera("camera_m4", offset,-offset,-object_depth-offset_depth, 0,0,0);
  environment->generateCamera("camera_m5", -offset,-offset,-object_depth-offset_depth, 0,0,0);
  environment->generateCamera("camera_m6", 0,-offset,-object_depth-offset_depth, 0,0,0);
  environment->generateCamera("camera_m7", -offset,0,-object_depth-offset_depth, 0,0,0);
  environment->generateCamera("camera_m8", offset,0,-object_depth-offset_depth, 0,0,0);
  
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

  renderer->renderImages_naive(environment);

  t_end=getTime();
  cout << "rendering took: " << (t_end-t_start) << " ms" << endl;
  // --------------------------------------

  Camera_cpu* camera_r = environment->camera_vector_cpu_[0];
  Image<float>* depth_map_gt = camera_r->depth_map_->clone("depth map gt");

  // clear depth maps
  for (Camera_cpu* camera : environment->camera_vector_cpu_){
    camera->depth_map_->setAllPixels(1.0);
    camera->depth_map_gpu_.setTo(cv::Scalar::all(1.0));
  }

  //############################################################################
  // compute depth map
  //############################################################################

  int num_cameras = environment->camera_vector_cpu_.size();
  for (int it=0; it<num_cameras; it++){
    // --------------------------------------
    // for the first camera, set it as the reference camera
    if (it==0){
      dtam->addCamera(environment->camera_vector_cpu_[it],environment->camera_vector_gpu_[it]);
      dtam->setReferenceCamera(it);
      continue;
    }

    // --------------------------------------
    cerr << "computing discrete cost volume " << it << "/" << num_cameras-1 << endl;
    t_start=getTime();

    dtam->addCamera(environment->camera_vector_cpu_[it],environment->camera_vector_gpu_[it]);
    dtam->prepareCameraForDtam(it);
    dtam->updateDepthMap_parallel_gpu(it);

    t_end=getTime();
    cerr << "discrete cost volume computation took: " << (t_end-t_start) << " ms " << it << "/" << num_cameras-1 << endl;
    // --------------------------------------


  }
  // --------------------------------------
  // show camera rgb images and depth maps
  for (Camera_cpu* camera : environment->camera_vector_cpu_){
    camera->image_rgb_->show(800/camera->resolution_);
    // camera->depth_map_->show(800/camera->resolution_);
  }
  camera_r->depth_map_gpu_.download(camera_r->depth_map_->image_);
  camera_r->depth_map_->show(800/camera_r->resolution_);
  depth_map_gt->show(800/camera_r->resolution_);
  cv::waitKey(0);
  // --------------------------------------
  return 1;
}
