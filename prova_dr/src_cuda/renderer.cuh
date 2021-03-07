#pragma once
#include "defs.h"
#include "image.h"
#include "environment.cuh"
#include "camera_cpu.cuh"
#include <cuda_runtime.h>

using namespace pr;


__global__ void renderPoint_gpu( Cp* cp, Camera_gpu* camera_gpu_d );
// __global__ void renderPoint_gpu(Cp& cp, Camera_cpu* camera );

class Renderer{
  public:

    bool renderPoint(Cp& cp, Camera_cpu* camera);
    void renderImage_naive(cpVector& cp_vector, Camera_cpu* camera);
    bool renderImages_parallel_gpu(Environment* environment);
};
