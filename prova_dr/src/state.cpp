#include "state.h"

void State::generateInitialGuess(){
  // Eigen::Vector3f p(0,0,0);
  // points_.push_back(p);
  for (int i=0; i<10; i++)
  {

  }
}

void State::generateWorldFrame(){
  // Eigen::Vector3f p(0,0,0);
  // points_.push_back(p);
  for (int i=0; i<10; i++)
  {
    float delta = (float)i/100;
    Eigen::Vector3f x(delta,0,0);
    cv::Vec3b color1(0,0,255);
    struct Cp cp1 = {x, color1};
    cps_world_frame.push_back(cp1);

    Eigen::Vector3f y(0,delta,0);
    cv::Vec3b color2(0,255,0);
    struct Cp cp2 = {y, color2};
    cps_world_frame.push_back(cp2);

    if (i>0)
    {
      Eigen::Vector3f z(0,0,delta);
      cv::Vec3b color3(255,0,0);
      struct Cp cp3 = {z, color3};
      cps_world_frame.push_back(cp3);
    }
  }
}
