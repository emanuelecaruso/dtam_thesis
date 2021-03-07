#pragma once
#include "defs.h"
#include "image.h"
#include "camera.h"

using namespace pr;

class Renderer{
  public:
    cpVector cp_vector_;

    Renderer(cpVector cp_vector){
      cp_vector_ = cp_vector;
    };

    bool renderPoint(Cp& cp, Camera* camera);
    bool renderImage_naive(cpVector& cp_vector, Camera* camera);
    bool renderImage_parallel_cpu(cpVector& cp_vector, Camera* camera);
};
