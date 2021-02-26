#include "renderer.cuh"
#include <thread>
#include <vector>
#include <mutex>

bool Renderer::renderPoint(Cp& cp, Camera* camera){

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

__global__ void renderPoint_gpu(Cp& cp, Camera* camera, bool& valid){

  // Eigen::Vector3f t_r(0,0,0);
  // Eigen::Isometry3f* frame_world_wrt_camera_r = new Eigen::Isometry3f;
  // frame_world_wrt_camera_r->linear().setIdentity();  //TODO implement orientation
  // frame_world_wrt_camera_r->translation()=t_r;
  // Eigen::Isometry3f* frame_camera_wrt_world_r = new Eigen::Isometry3f;
  // *frame_camera_wrt_world_r = frame_world_wrt_camera_r->inverse();
  // Camera* camera_ = new CameraGPU("name",0,0,0,0,0,frame_camera_wrt_world_r,frame_world_wrt_camera_r);

  
  // Eigen::Vector2f uv;
  // float depth_cam;
  // bool point_in_front_of_camera = camera->projectPoint(cp.point, uv, depth_cam );
  // if (!point_in_front_of_camera)
  //   valid = false;

  // float width = camera->width_;
  // float height = camera->width_/camera->aspect_;
  //
  // if(uv.x()<0 || uv.x()>width)
  //   valid = false;
  // if(uv.y()<0 || uv.y()>height)
  //   valid = false;
  //
  // Eigen::Vector2i pixel_coords;
  // camera->uv2pixelCoords( uv, pixel_coords);
  //
  // float depth = depth_cam/camera->max_depth_;
  //
  // float evaluated_pixel;
  // camera->depth_map_->evalPixel(pixel_coords,evaluated_pixel);
  //
  // if (evaluated_pixel<depth)
  //   valid = false;
  //
  // if (depth>1 || depth>255 || cp.color[0]>255 || cp.color[1]>255 || cp.color[2]>255)
  //   valid = false;
  //
  // cv::Vec3b color = cv::Vec3b(cp.color[0],cp.color[1],cp.color[2]);
  //
  // camera->image_rgb_->setPixel(pixel_coords, color);
  // camera->depth_map_->setPixel(pixel_coords,depth);
  //
  // valid = true;
}

void Renderer::renderImage_naive(cpVector& cp_vector, Camera* camera){

    camera->clearImgs();
    for (Cp cp : cp_vector)
    {
      Renderer::renderPoint(cp, camera);
    }

}

void Renderer::renderImage_parallel_gpu(cpVector& cp_vector, Camera* camera){

  camera->clearImgs();


}
