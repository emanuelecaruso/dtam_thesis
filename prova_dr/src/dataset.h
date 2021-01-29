#pragma once
#include <string>
#include "json.hpp"
#include "camera.h"
#include "image.h"

/**
   Dataset class.
   Supports simple data extraction operations, given the path
*/
using namespace pr;
using json = nlohmann::json;

class Dataset{
  public:
    std::string path_;
    Dataset(std::string path){
      path_ = path;
    };

    // function that extract json given its path
    bool collectCameras(CameraVector& camera_vector, float max_depth);
    bool collectImages(std::vector< Image<cv::Vec3b>* >& image_vector);

  private:
    bool load_json(json& js);
    void extractImgNames(std::vector<std::string>& v);


};
