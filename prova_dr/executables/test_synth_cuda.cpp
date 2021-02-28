#include "defs.h"
#include "dataset.h"
#include "camera_cpu.cuh"
#include "image.h"
#include "renderer.cuh"
#include "environment.cuh"
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

  cudaError_t err ;

  double t_start=getTime();  // time start for computing computation time
  double t_end=getTime();    // time end for computing computation time

  cpVector cp_vector; // vector of colored points populating the world
  CameraVector_cpu camera_vector; // vector containing pointers to camera objects

  Environment* environment = new Environment(); // environment generator object (pointer)
  Renderer* renderer = new Renderer(); // renderer object (pointer)
  // Dtam* dtam = new Dtam(1); // dense mapper and tracker

  //############################################################################
  // generate 2 cameras (in this case same orientation, shift on x axis)
  //############################################################################

  // --------------------------------------
  // generate cameras
  float object_depth=2;

  Camera_cpu* camera_r = environment->generateCamera("camera_r", 0,0,-object_depth, 0,0,0);
  Camera_cpu* camera_m1 = environment->generateCamera("camera_m1", 0.1,0.1,-object_depth-0.1, 0,0,0);
  Camera_cpu* camera_m2 = environment->generateCamera("camera_m2", -0.1,-0.1,-object_depth-0.1, 0,0,0);

  camera_vector.push_back(camera_r);
  camera_vector.push_back(camera_m1);
  camera_vector.push_back(camera_m2);

  Camera_gpu* camera_r_d = camera_r->getCamera_gpu();
  Camera_gpu* camera_m1_d = camera_m1->getCamera_gpu();
  Camera_gpu* camera_m2_d = camera_m2->getCamera_gpu();

  // --------------------------------------
  // generate environment
  cout << "generating environment.." << endl;
  t_start=getTime();

  int density=6000; // NOTE: to work cp_vectorwith cuda, density^2 must be a multiplier of 32
  environment->generateSinusoidalSurface(object_depth, density, cp_vector);
  Cp* cpVector_gpu = environment->getCpPtrOnGPU(cp_vector);


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

  // for (Camera_cpu* camera_cpu : camera_vector){
  //   Camera_gpu* camera_gpu = camera_cpu->getCamera_gpu();
  //   renderer->renderImage_parallel_gpu(cp_vector, cp_vector.size(), camera_gpu);
  //   camera_cpu->gpuFree();
  //   delete camera_gpu;
  // }
  // std::cout << cp_vector[0].point << std::endl;

  renderer->renderImage_parallel_gpu(cpVector_gpu, cp_vector.size(), camera_r_d, camera_r);
  renderer->renderImage_parallel_gpu(cpVector_gpu, cp_vector.size(), camera_m1_d, camera_m1);
  renderer->renderImage_parallel_gpu(cpVector_gpu, cp_vector.size(), camera_m2_d, camera_m2);
  // camera_cpu->gpuFree();
  // delete camera_gpu;

    // renderer->renderImage_naive(cp_vector, camera);

  t_end=getTime();
  cout << "rendering took: " << (t_end-t_start) << " ms" << endl;
  // --------------------------------------

  Image<float>* depth_map_gt = camera_r->depth_map_->clone("depth map gt");
  // Image<cv::Vec3b>* rgb_image_m_gt = camera_m1->image_rgb_->clone("rgb image m gt");

  // clear depth maps
  for (Camera_cpu* camera : camera_vector)
    camera->depth_map_->image_=1.0;




  //
  // //############################################################################
  // // compute depth map
  // //############################################################################
  //
  // // Eigen::Vector2i pixel_coords_r(50,50);
  // Dtam* dtam = new Dtam(camera_vector);
  //
  // cout << "computing discrete cost volume..." << endl;
  // t_start_projection=getTime();
  //
  // dtam->getDepthMap(100);
  // // dtam->getDepthMap(camera_vector, 100, true);
  // // dtam->getDepthMap(camera_vector, 0.25);
  // // dtam->getDepthMap(camera_vector, 0.25, true);
  //
  // t_end_projection=getTime();
  // cout << "discrete cost volume computation took: " << (t_end_projection-t_start_projection) << " ms" << endl;
  //
  //
  for (Camera_cpu* camera : camera_vector){
    camera->image_rgb_->show(800/camera->resolution_);
    camera->depth_map_->show(800/camera->resolution_);
  }
  depth_map_gt->show(800/camera_vector[0]->resolution_);
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
