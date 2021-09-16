#include "camera_cpu.cuh"
#include <thread>
#include <vector>
#include <mutex>

using namespace std;
using namespace pr;



void Camera_cpu::gpuFree(){
  image_rgb_gpu_.release();
  depth_map_gpu_.release();
}


Camera_gpu* Camera_cpu::getCamera_gpu(){

  image_rgb_gpu_.upload(image_rgb_->image_);
  depth_map_gpu_.upload(depth_map_->image_);

  Camera_gpu* camera_gpu_h = new Camera_gpu(name_, lens_, aspect_, width_, resolution_,
     max_depth_, K_, Kinv_, *frame_camera_wrt_world_, *frame_world_wrt_camera_,
      depth_map_gpu_, image_rgb_gpu_);

  cudaError_t err ;

  Camera_gpu* camera_gpu_d;
  cudaMalloc((void**)&camera_gpu_d, sizeof(Camera_gpu));
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("cudaMalloc %s%s",name_," Error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(camera_gpu_d, camera_gpu_h, sizeof(Camera_gpu), cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("cudaMemcpy %s%s",name_," Error: %s\n", cudaGetErrorString(err));

  delete camera_gpu_h;

  return camera_gpu_d;
}

void Camera_cpu::cloneCameraImages(Camera* camera){
  depth_map_ = camera->depth_map_;
  image_rgb_ = camera->image_rgb_;

}

void Camera_cpu::showInvdepthmap(int scale){
  Image< float >* invdepthmap = new Image< float >;
  camera_r->depth_map_gpu_.download(invdepthmap->image_);
  invdepthmap->image_=1.0/(2.0*(invdepthmap->image_));
  invdepthmap->show(scale/resolution_);
}
