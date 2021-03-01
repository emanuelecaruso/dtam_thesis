#include "renderer.cuh"
#include <thread>
#include <vector>
#include <mutex>

bool Renderer::renderPoint(Cp& cp, Camera_cpu* camera){

  Eigen::Vector2f uv;
  float depth_cam;
  bool point_in_front_of_camera = camera->projectPoint(cp.point, uv, depth_cam );
  if (!point_in_front_of_camera)
    return false;

  float width = camera->width_;
  float height = camera->width_/camera->aspect_;

  if(uv.x()<0 || uv.x()>width)
    return false;
  if(uv.y()<0 || uv.y()>height)
    return false;

  Eigen::Vector2i pixel_coords;
  camera->uv2pixelCoords( uv, pixel_coords);

  float depth = depth_cam/camera->max_depth_;

  float evaluated_pixel;
  camera->depth_map_->evalPixel(pixel_coords,evaluated_pixel);

  if (evaluated_pixel<depth)
    return false;

  if (depth>1 || depth>255 || cp.color[0]>255 || cp.color[1]>255 || cp.color[2]>255)
    return false;

  cv::Vec3b color = cv::Vec3b(cp.color[0],cp.color[1],cp.color[2]);

  camera->image_rgb_->setPixel(pixel_coords, color);
  camera->depth_map_->setPixel(pixel_coords,depth);

  return true;
}

__global__ void renderPoint_gpu(Cp* cp, Camera_gpu* camera_gpu_d ){
// __global__ void renderPoint_gpu( ){

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  bool valid = true;
  Eigen::Vector2f uv; float depth_cam;

  bool point_in_front_of_camera = camera_gpu_d->projectPoint(cp[i].point, uv, depth_cam );
  if (!point_in_front_of_camera)
    valid = false;

  float width = camera_gpu_d->width_;
  float height = camera_gpu_d->width_/camera_gpu_d->aspect_;

  if(uv.x()<0 || uv.x()>width)
    valid = false;
  if(uv.y()<0 || uv.y()>height)
    valid = false;

  Eigen::Vector2i pixel_coords;
  camera_gpu_d->uv2pixelCoords( uv, pixel_coords);

  float depth = depth_cam/camera_gpu_d->max_depth_;

  // printf("\n");
  // printf("%f",cp[i].point.x());
  // printf("%f",cp[i].point.y());
  // printf("%f",cp[i].point.z());
  // printf("\n");

  float evaluated_pixel = camera_gpu_d->depth_map_(pixel_coords.y(),pixel_coords.x());

  if (evaluated_pixel<depth)
    valid = false;

  if (depth>1 || depth>255 || cp[i].color[0]>255 || cp[i].color[1]>255 || cp[i].color[2]>255)
    valid = false;

  if (valid){
    uchar3 color = make_uchar3( cp[i].color[0], cp[i].color[1], cp[i].color[2] );
    camera_gpu_d->image_rgb_(pixel_coords.y(),pixel_coords.x())= color;
    camera_gpu_d->depth_map_(pixel_coords.y(),pixel_coords.x())= depth;
  }
}

void Renderer::renderImage_naive(cpVector& cp_vector, Camera_cpu* camera){

    camera->clearImgs();
    for (Cp cp : cp_vector)
    {
      Renderer::renderPoint(cp, camera);
    }

}
bool Renderer::renderImages_parallel_gpu(Environment* environment){

  cudaError_t err ;
  Cp* cp_d = environment->cp_d_;
  int cp_size = environment->cp_vector_.size();

  int numThreads = 32;
  int numBlocks = cp_size / numThreads;

  if (cp_size % numThreads != 0)
    return false;

  for (int i=0; i<environment->camera_vector_cpu_.size(); i++){
    Camera_cpu* camera_cpu = environment->camera_vector_cpu_[i];
    Camera_gpu* camera_gpu = environment->camera_vector_gpu_[i];


    renderPoint_gpu<<<numBlocks,numThreads>>>( cp_d, camera_gpu );
    // renderPoint_gpu<<<1,1>>>( cp_d, camera_gpu_d );
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error executing rendering kernel: %s\n", cudaGetErrorString(err));

    // auto a = camera_cpu->image_rgb_gpu_;

    camera_cpu->image_rgb_gpu_.download(camera_cpu->image_rgb_->image_);
    camera_cpu->depth_map_gpu_.download(camera_cpu->depth_map_->image_);
    // camera_cpu->image_rgb_gpu_.download(image_rgb_->image_);
    // cudaMemcpy(valid_h, valid_d, sizeof(bool), cudaMemcpyDeviceToHost);
    // std::cout << *valid_h << std::endl;

  }



  return true;

}
