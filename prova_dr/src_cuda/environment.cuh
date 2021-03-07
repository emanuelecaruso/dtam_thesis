#pragma once
#include "defs.h"
#include "camera_cpu.cuh"
#include "camera_gpu.cuh"


class Environment{
  public:
    // cameras parameters
    int resolution_;
    float film_;
    float lens_;
    float aspect_;
    float max_depth_;
    // environment state
    cpVector cp_vector_;
    Cp* cp_d_;
    CameraVector_cpu camera_vector_cpu_; // vector containing pointers to camera objects
    CameraVector_gpu camera_vector_gpu_; // vector containing pointers to camera objects on device (gpu)


    Environment(int resolution = 640, float film = 0.024, float lens = 0.035,
                          float aspect = 1, float max_depth =4.2)
        {
          resolution_=resolution; film_=film; lens_=lens;
          aspect_=aspect; max_depth_=max_depth;
        }

    void generateSinusoidalSurface(float picks_depth, int density);
    void generateCamera(std::string name, float t1, float t2, float t3, float alpha1, float alpha2, float alpha3);
  private:
    Cp* getCpPtrOnGPU();
};
