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

using namespace std;
using namespace pr;


int main (int argc, char * argv[]) {


  //############################################################################
  // initialization
  //############################################################################
  std::string dataset_name = argv[1]; // dataset name

  std::cout << "dataset name: " << dataset_name << std::endl;

  double t_start=getTime();  // time start for computing computation time
  double t_end=getTime();    // time end for computing computation time

  Environment_gpu* environment = new Environment_gpu(); // environment generator object (pointer)
  Dtam* dtam = new Dtam(environment); // dense mapper and tracker


  // std::string dataset_name = "rotatedcube_9cams_64res"; // dataset name

  // std::string dataset_name = "rotatedcube_2cams"; // dataset name
  // std::string dataset_name = "rotatedcube_9cams"; // dataset name
  // std::string dataset_name = "rotatedcube_17cams"; // dataset name
  // std::string dataset_name = "rotatedcube_25cams"; // dataset name
  // std::string dataset_name = "sin_9cams"; // dataset name
  // std::string dataset_name = "sin_9cams_64res"; // dataset name

  std::string path_name = "./dataset/"+dataset_name; // dataset name

  environment->loadEnvironment_gpu(path_name, dataset_name);


  Camera_cpu* camera_r = environment->camera_vector_cpu_[0];
  Image<float>* depth_map_gt = camera_r->depth_map_->clone("depth map gt");

  // // clear depth maps
  // for (Camera_cpu* camera : environment->camera_vector_cpu_){
  //   camera->depth_map_->setAllPixels(1.0);
  //   camera->depth_map_gpu_.setTo(cv::Scalar::all(1.0));
  // }


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
    // cerr << "computing discrete cost volume " << it << "/" << num_cameras-1 << endl;
    // t_start=getTime();

    dtam->addCamera(environment->camera_vector_cpu_[it],environment->camera_vector_gpu_[it]);
    dtam->prepareCameraForDtam(it);
    dtam->updateDepthMap_parallel_gpu(it);

    // t_end=getTime();
    // cerr << "discrete cost volume computation took: " << (t_end-t_start) << " ms " << it << "/" << num_cameras-1 << endl;
    // --------------------------------------

    // show final depthmap
    // camera_r->depth_map_gpu_.download(camera_r->depth_map_->image_);
    // camera_r->depth_map_->show(800/camera_r->resolution_);
    // depth_map_gt->show(800/camera_r->resolution_);
    camera_r->showInvdepthmap(800);
    cv::waitKey(0);

  }
  cv::waitKey(0);

  // --------------------------------------
  // show camera rgb images and depth maps
  // for (Camera_cpu* camera : environment->camera_vector_cpu_){
    // camera->image_rgb_->show(800/camera->resolution_);
    // camera->depth_map_->show(800/camera->resolution_);
  // }


  // // --------------------------------------
  return 1;
}
