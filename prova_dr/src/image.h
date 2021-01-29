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

    inline void show(int image_scale=1){
      cv::Mat_< T > resized_image;
      cv::resize(image_, resized_image, cv::Size(), image_scale, image_scale, CV_INTER_NN);
      cv::imshow(name_, resized_image);
    }


    inline Image* clone(){
      return new Image(*this);
    }

    inline bool evalPixel(Eigen::Vector2i& uv, T& color){
      if (uv.y()>=0 && uv.y()<image_.rows && uv.x()>=0 && uv.x()<image_.cols)
      {
        color = image_.template at<T>(uv.y(),uv.x());
        return true;
      }
      return false;
    }

    inline bool setPixel(Eigen::Vector2i& uv, T color){
      if (uv.y()>=0 && uv.y()<image_.rows && uv.x()>=0 && uv.x()<image_.cols)
      {
        image_.template at<T>(uv.y(),uv.x()) = color;
        return true;
      }
      return false;
    }

    inline bool test(Eigen::Vector2i& i, T c){

      return false;
    }
};

using ImageVector = std::vector< Image<cv::Vec3b>* >;
