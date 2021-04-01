#include "image.h"

int mseBetween2Colors(cv::Vec3b& clr1, cv::Vec3b& clr2){

  // L2 norm
  int mse=(clr1(0)-clr2(0))*(clr1(0)-clr2(0))+(clr1(1)-clr2(1))*(clr1(1)-clr2(1))+(clr1(2)-clr2(2))*(clr1(2)-clr2(2));

  // // L1 norm / 3
  // int mse=(abs(clr1(0)-clr2(0))+abs(clr1(1)-clr2(1))+abs(clr1(2)-clr2(2)))/3;

  return mse;
}
