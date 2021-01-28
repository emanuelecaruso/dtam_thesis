#pragma once
#include "defs.h"

using namespace pr;

template<class T>
class Image{
  public:
    std::string name_;
    cv::Mat_< T > image_;
    Image(std::string name){
      name_ = name;
    };

    inline void loadJpg(std::string path){
      image_ = cv::imread(path);
      if(image_.empty())
      std::cout << "Could not read the image: " << path << std::endl;
    }

    inline void show(){cv::imshow(name_, image_);}

    inline void clear(){
      image_= T(255);
    }

    inline bool evalPixel(Eigen::Vector2i& uv, T& color){
      if (uv[0]>=0 && uv[0]<image_.cols && uv[1]>=0 && uv[1]<image_.rows)
      {
        color = image_.template at<T>(uv[0],uv[1]);
        return true;
      }
      return false;
    }

    inline bool setPixel(Eigen::Vector2i& uv, T color){
      if (uv[0]>=0 && uv[0]<image_.cols && uv[1]>=0 && uv[1]<image_.rows)
      {
        image_.template at<T>(uv[0],uv[1]) = color;
        return true;
      }
      return false;
    }

    inline bool test(Eigen::Vector2i& i, T c){

      return false;
    }
};

using ImageVector = std::vector< Image<cv::Vec3b>* >;
