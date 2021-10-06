#pragma once
#include "camera_cpu.cuh"
#include "camera_gpu.cuh"
#include "image.h"
#include <cuda_runtime.h>
#include "environment.cuh"
#include "mapper.cuh"


class Dtam{

  public:
    CameraVector_cpu camera_vector_cpu_;
    CameraVector_gpu camera_vector_gpu_;
    Mapper* mapper_;
    int index_r_;
    float* invdepth_r_array_;



    Dtam(Environment_gpu* environment){
      mapper_ = new Mapper(environment);

    };

    void test_mapping(Environment_gpu* environment);
    void test_tracking(Environment_gpu* environment);
    
    void showImgs(int scale);

    void addCamera(Camera_cpu* camera_cpu, Camera_gpu* camera_gpu);

};
