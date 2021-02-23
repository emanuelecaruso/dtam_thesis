#include "dataset.h"
#include "image.h"
#include <iostream>


bool Dataset::load_json(json& js){
  std::string text="";
  std::string error="";

  std::string scene_name = path_.substr(10, path_.size()-1);
  std::string path_json = path_+"/"+scene_name+".json";

  // https://stackoverflow.com/questions/174531/how-to-read-the-content-of-a-file-to-a-string-in-c
  auto fs = fopen(path_json.c_str(), "rb");
  if (!fs)
  {
    error = path_ + ": file not found";
    std::cout << error << std::endl;
    return false;
  }
  fseek(fs, 0, SEEK_END);
  auto length = ftell(fs);
  fseek(fs, 0, SEEK_SET);
  text.resize(length);
  if (fread(&text[0], 1, length, fs) != length)
  {
    error = path_ + ": read error";
    std::cout << error << std::endl;
    return false;
  }


  try {
    js = json::parse(text);
  } catch (std::exception& e) {
    {
      error = path_ + ": parse error in json";
      std::cout << error << std::endl;
      return false;
    };
  }
  return true;
}

bool Dataset::collectCameras(CameraVector& camera_vector, float max_depth){
  json js;
  if (!Dataset::load_json(js)) return false;
  auto cameras = js.at("cameras");

  for (json::iterator it = cameras.begin(); it != cameras.end(); ++it) {
    std::string name=it.key();
    float lens;
    float aspect;
    float width;
    int resolution;
    nlohmann::basic_json<>::value_type f;
    try{
      lens = it.value().at("lens");
      aspect = it.value().at("aspect");
      width = it.value().at("width");
      resolution = it.value().at("resolution");
      f= it.value().at("frame");
    } catch (std::exception& e) {
      std::string error = ": missing values in json file";
      std::cout << error << std::endl;
      return false;
    };

    Eigen::Matrix3f R;
    R <<
      f[0], f[1], f[2],
      f[3], f[4], f[5],
      f[6], f[7], f[8];


    Eigen::Vector3f t(f[9],f[10],f[11]);
    Eigen::Isometry3f frame_camera_wrt_world;
    frame_camera_wrt_world.linear()=R;
    frame_camera_wrt_world.translation()=t;
    Eigen::Isometry3f frame_world_wrt_camera;
    frame_world_wrt_camera=frame_camera_wrt_world.inverse();


    Camera* cam = new Camera(name,lens,aspect,width,resolution,max_depth,&frame_camera_wrt_world,&frame_world_wrt_camera);
    camera_vector.push_back(cam);
  }

  return true;
}

void Dataset::extractImgNames(std::vector<std::string>& v){
    DIR* dirp = opendir(path_.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        std::string name=dp->d_name;
        int l=name.size();
        if (l>3){
          std::string sbstr = name.substr(l-4,l-1);
          if (sbstr==".jpg")
            v.push_back(name);
        }
        std::sort (v.begin(), v.end());
    }
    closedir(dirp);
}

bool Dataset::collectImages(std::vector<Image<cv::Vec3b>*>& image_vector){
  std::vector<std::string> img_names;
  Dataset::extractImgNames(img_names);
  for (std::string img_name : img_names)
  {
    Image<cv::Vec3b>* img = new Image<cv::Vec3b>(img_name);
    img->loadJpg(path_+"/"+img_name);
    image_vector.push_back(img);
  }
  return true;
}
