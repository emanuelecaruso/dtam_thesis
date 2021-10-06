#pragma once
#include "defs.h"
#include "environment.cuh"
#include "camera_cpu.cuh"
#include <cuda_runtime.h>


class Tracker{

  public:
    Tracker(Environment_gpu* environment){};

    int index_r_;
    CameraVector_cpu camera_vector_cpu_;
    CameraVector_gpu camera_vector_gpu_;
};
