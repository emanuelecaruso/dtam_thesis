#pragma once
#include "defs.h"
#include "image.h"
#include "camera_cpu.cuh"
#include <cuda_runtime.h>

using namespace pr;

struct Cp // Colored point (in 3D)
{
  Eigen::Vector3f point;
  int color[3];
};

typedef std::vector<Cp> cpVector;

__global__ void renderPoint_gpu( Cp* cp, Camera_gpu* camera );
// __global__ void renderPoint_gpu(Cp& cp, Camera_cpu* camera );

class Renderer{
  public:

    bool renderPoint(Cp& cp, Camera_cpu* camera);
    void renderImage_naive(cpVector& cp_vector, Camera_cpu* camera);
    bool renderImage_parallel_gpu(Cp* cp_d, int cp_size, Camera_gpu* camera_gpu_d, Camera_cpu* camera_cpu);
};
