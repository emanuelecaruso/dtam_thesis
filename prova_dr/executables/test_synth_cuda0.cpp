#include "defs.h"
#include "dataset.h"
#include "camera_cpu.cuh"
#include "image.h"
#include "environment.cuh"
#include "environment.h"
#include "renderer.cuh"
#include "utils.h"
#include "dtam_cuda.cuh"
#include <stdio.h>
#include <unistd.h>

using namespace std;
using namespace pr;


int main (int argc, char * argv[]) {

  //############################################################################
  // choose parameters
  //############################################################################

  int resolution = 640;
  float aspect = 1.333333333;
  // float aspect = 1;

  //############################################################################
  // initialization
  //############################################################################

  double t_start=getTime();  // time start for computing computation time
  double t_end=getTime();    // time end for computing computation time

  Environment_gpu* environment = new Environment_gpu(resolution, aspect); // environment generator object (pointer)
  Dtam* dtam = new Dtam(environment); // dense mapper and tracker
  Renderer* renderer = new Renderer(); // renderer object (pointer)

  //############################################################################
  // generate 2 cameras (in this case same orientation, shift on x axis)
  //############################################################################

  // --------------------------------------

  // generate cameras
  float object_depth=2;
  float offset=-0.1;
  float offset_depth=-0.05;

  environment->generateCamera("camera0", 0,0,-object_depth, 0,0,0);
  environment->generateCamera("camera1", offset,offset,-object_depth-offset_depth, 0,0,0);
  // environment->generateCamera("camera2", -offset,offset,-object_depth-offset_depth, 0,0,0);
  // environment->generateCamera("camera3", 0,offset,-object_depth-offset_depth, 0,0,0);
  // environment->generateCamera("camera4", offset,-offset,-object_depth-offset_depth, 0,0,0);
  // environment->generateCamera("camera5", -offset,-offset,-object_depth-offset_depth, 0,0,0);
  // environment->generateCamera("camera6", 0,-offset,-object_depth-offset_depth, 0,0,0);
  // environment->generateCamera("camera7", -offset,0,-object_depth-offset_depth, 0,0,0);
  // environment->generateCamera("camera8", offset,0,-object_depth-offset_depth, 0,0,0);
  // --------------------------------------
  // generate environment
  cerr << "generating environment.." << endl;
  t_start=getTime();

  int density=5000;
  Eigen::Isometry3f pose_cube;
  // pose.linear().setIdentity();
  pose_cube.linear()=Rx(M_PI/6)*Ry(M_PI/4)*Rz(3.14/6);
  pose_cube.translation()= Eigen::Vector3f(0,0,-1);
  environment->generateTexturedCube(1, pose_cube, density);

  Eigen::Isometry3f pose_background;
  pose_background.linear().setIdentity();
  pose_background.translation()= Eigen::Vector3f(0,0,-1.8);
  environment->generateTexturedPlane("images/sunshine.jpg", 4, pose_background, density);

  // environment->generateSinusoidalSurface(object_depth, density);
  // environment->generateTexturedPlane("images/leon.jpg", 1, pose, density);

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

  renderer->renderImages_naive(environment);

  t_end=getTime();
  cerr << "rendering took: " << (t_end-t_start) << " ms" << endl;
  // --------------------------------------


  Camera_cpu* camera_r = environment->camera_vector_cpu_[0];
  Image<float>* depth_map_gt = camera_r->depth_map_->clone("depth map gt");

  // clear depth maps
  for (Camera_cpu* camera : environment->camera_vector_cpu_){
    camera->printMembers();
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
    camera->image_rgb_->show(800/camera->resolution_, "_ori");
    camera->depth_map_->show(800/camera->resolution_, "_ori");
  }
  // show final depthmap
  camera_r->depth_map_gpu_.download(camera_r->depth_map_->image_);
  camera_r->depth_map_->show(800/camera_r->resolution_, "_ori");
  depth_map_gt->show(800/camera_r->resolution_, "_ori");


  // environment->saveEnvironment( path_name, dataset_name);


  std::string dataset_name = "rotatedcube_2cams"; // dataset name
  std::string path_name = "./dataset/"+dataset_name; // dataset name


  environment->loadEnvironment_gpu( path_name, dataset_name);
  Camera_cpu* camera_r_ = environment->camera_vector_cpu_[0];
  Image<float>* depth_map_gt_ = camera_r_->depth_map_->clone("depth map gt");

  // clear depth maps
  for (Camera_cpu* camera : environment->camera_vector_cpu_){
    camera->printMembers();
    camera->depth_map_->setAllPixels(1.0);
    camera->depth_map_gpu_.setTo(cv::Scalar::all(1.0));
  }

  //############################################################################
  // compute depth map
  //############################################################################



  Dtam* dtam_ = new Dtam(environment); // dense mapper and tracker


  int num_cameras_ = environment->camera_vector_cpu_.size();
  for (int it=0; it<num_cameras_; it++){
    // --------------------------------------
    // for the first camera, set it as the reference camera
    if (it==0){
      dtam_->addCamera(environment->camera_vector_cpu_[it],environment->camera_vector_gpu_[it]);
      dtam_->setReferenceCamera(it);
      continue;
    }

    // --------------------------------------
    cerr << "computing discrete cost volume " << it << "/" << num_cameras_-1 << endl;
    t_start=getTime();

    dtam_->addCamera(environment->camera_vector_cpu_[it],environment->camera_vector_gpu_[it]);
    dtam_->prepareCameraForDtam(it);
    dtam_->updateDepthMap_parallel_gpu(it);

    t_end=getTime();
    cerr << "discrete cost volume computation took: " << (t_end-t_start) << " ms " << it << "/" << num_cameras_-1 << endl;
    // --------------------------------------


  }
  // --------------------------------------


  // show camera rgb images and depth maps
  for (Camera_cpu* camera : environment->camera_vector_cpu_){
    camera->image_rgb_->show(800/camera->resolution_, "_after");
    camera->depth_map_->show(800/camera->resolution_, "_after");
  }
   // show final depthmap
  camera_r_->depth_map_gpu_.download(camera_r_->depth_map_->image_);
  camera_r_->depth_map_->show(800/camera_r_->resolution_, "_after");
  depth_map_gt_->show(800/camera_r_->resolution_, "_after");


  cv::waitKey(0);
  // // --------------------------------------
  return 1;
}
