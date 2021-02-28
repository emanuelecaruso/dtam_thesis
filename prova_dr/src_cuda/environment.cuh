#pragma once
#include "renderer.cuh"
#include "camera_cpu.cuh"

class Environment{
  public:
    // camera parameters
    int resolution_;
    float film_;
    float lens_;
    float aspect_;
    float max_depth_;


    Environment(int resolution = 600, float film = 0.024, float lens = 0.035,
                          float aspect = 1, float max_depth =4.2)
        {
          resolution_=resolution; film_=film; lens_=lens;
          aspect_=aspect; max_depth_=max_depth;
        }

    void generateSinusoidalSurface(float picks_depth, int density, cpVector& cp_vector);
    Camera_cpu* generateCamera(std::string name, float t1, float t2, float t3, float alpha1, float alpha2, float alpha3);

    Cp* getCpPtrOnGPU(cpVector& cp_vector);
};
