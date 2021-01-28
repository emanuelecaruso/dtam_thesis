#pragma once
#include "defs.h"

using namespace pr;

class State{
  public:
    cpVector cp_vector;
    cpVector cps_world_frame;
    Eigen::Vector3f location_;
    State(Eigen::Vector3f location){
      location_ = location;
    };

    void generateInitialGuess();
    void generateWorldFrame();
};
