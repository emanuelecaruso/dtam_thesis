#include "environment.cuh"
#include "utils.h"

void Environment::generateSinusoidalSurface(float picks_depth, int density){

  // generate a "super dense" cloud of points expressed in camera_r frame
  float left_bound=-picks_depth/3-(0.1*picks_depth);
  float right_bound=(picks_depth/3)+(0.1*picks_depth);

  for (int x=0; x<density; x++)
    for (int y=0; y<density; y++){
      float x_ = ((float)x/(float)density)*(right_bound-left_bound)+left_bound;
      float y_ = ((float)y/(float)density)*(-left_bound-left_bound)+left_bound;

      float depth = ((sin((x_)*(6*3.14))*sin((x_)*(6*3.14))+sin((y_)*(6*3.14))*sin((y_)*(6*3.14)))/2.0);

      // int clr_x = ((float)x/(float)density)*255*(sin((x_)*(6*3.14))*sin((x_)*(6*3.14)));
      // int clr_y = ((float)y/(float)density)*255*(sin((y_)*(6*3.14))*sin((y_)*(6*3.14)));
      // int clr_z = depth*(255.0/picks_depth);

      unsigned char clr_x = ((float)x/(float)density)*255*depth;
      unsigned char clr_y = ((float)y/(float)density)*255*depth;
      unsigned char clr_z = depth*(255.0/picks_depth);


      Cp cp;
      cp.point=Eigen::Vector3f(x_,y_,depth);
      cp.color[0]=clr_x;
      cp.color[1]=clr_y;
      cp.color[2]=clr_z;
      cp_vector_.push_back(cp);
    }

  cp_d_ = Environment::getCpPtrOnGPU();

}

Cp* Environment::getCpPtrOnGPU(){

  cudaError_t err ;

  Cp* cp_vector_h = &cp_vector_[0];
  Cp* cp_vector_d;

  cudaMalloc(&cp_vector_d, sizeof(Cp)*cp_vector_.size());
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("cudaMalloc cp Error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(cp_vector_d, cp_vector_h, sizeof(Cp)*cp_vector_.size(), cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("cudaMemcpy cp Error: %s\n", cudaGetErrorString(err));

  return cp_vector_d;
}


void Environment::generateCamera(std::string name, float t1, float t2, float t3, float alpha1, float alpha2, float alpha3){
  Eigen::Vector3f t_r(t1,t2,t3);
  Eigen::Isometry3f* frame_world_wrt_camera_r = new Eigen::Isometry3f;
  frame_world_wrt_camera_r->linear().setIdentity();  //TODO implement orientation
  frame_world_wrt_camera_r->translation()=t_r;
  Eigen::Isometry3f* frame_camera_wrt_world_r = new Eigen::Isometry3f;
  *frame_camera_wrt_world_r = frame_world_wrt_camera_r->inverse();
  Camera_cpu* camera = new Camera_cpu(name,lens_,aspect_,film_,resolution_,max_depth_,frame_camera_wrt_world_r,frame_world_wrt_camera_r);
  camera_vector_cpu_.push_back(camera);
  Camera_gpu* camera_d = camera->getCamera_gpu();
  camera_vector_gpu_.push_back(camera_d);

}
