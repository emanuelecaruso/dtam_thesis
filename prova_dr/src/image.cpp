#include "image.h"

int mseBetween2Colors(cv::Vec3b& clr1, cv::Vec3b& clr2){
  int mse=(pow((int)clr1(0)-(int)clr2(0),2)+pow((int)clr1(1)-(int)clr2(1),2)+pow((int)clr1(2)-(int)clr2(2),2));
  return mse;
}
