#include "environment.h"
#include "json.hpp"
#include "utils.h"
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
using json = nlohmann::json;

void Environment::generateLine(int density){
  for (int z=0; z<density; z++){
    float z_ = -((float)z/(float)density)*19-1;

    Cp cp;
    cp.point=Eigen::Vector3f(0,0,z_);
    cp.color[0]=0;
    cp.color[1]=255;
    cp.color[2]=0;
    cp_vector_.push_back(cp);
  }
}


void Environment::generateSinusoidalSurface(float picks_depth, int density){

  // generate a "super dense" cloud of points expressed in camera_r frame
  float left_bound=-picks_depth/3-(0.1*picks_depth);
  float right_bound=(picks_depth/3)+(0.1*picks_depth);

  float x_, y_;
  for (int x=0; x<density; x++){
    x_ = ((float)x/(float)density)*(right_bound-left_bound)+left_bound;
    for (int y=0; y<density; y++){
      y_ = ((float)y/(float)density)*(-left_bound-left_bound)+left_bound;

      float depth = ((sin((x_)*(6*3.14))*sin((x_)*(6*3.14))+sin((y_)*(6*3.14))*sin((y_)*(6*3.14)))/2.0);

      // int clr_x = ((float)x/(float)density)*255*(sin((x_)*(6*3.14))*sin((x_)*(6*3.14)));
      // int clr_y = ((float)y/(float)density)*255*(sin((y_)*(6*3.14))*sin((y_)*(6*3.14)));
      // int clr_z = depth*(255.0/picks_depth);

      int clr_x = ((float)x/(float)density)*255*depth;
      int clr_y = ((float)y/(float)density)*255*depth;
      int clr_z = depth*(255.0/picks_depth);


      Cp cp;
      cp.point=Eigen::Vector3f(x_,y_,depth);
      cp.color[0]=clr_x;
      cp.color[1]=clr_y;
      cp.color[2]=clr_z;
      cp_vector_.push_back(cp);
    }
  }
}

void Environment::generateTexturedPlane(std::string path, float size, Eigen::Isometry3f pose, int density){

  Image<cv::Vec3b>* img = new Image<cv::Vec3b>(path);
  img->loadJpg(path);

  float x_, y_, ratio_x, ratio_y;
  for (int x=0; x<density; x++){
    ratio_x = (float)x/(float)density;
    x_ = (-size/2)+ratio_x*(size);
    for (int y=0; y<density; y++){
      ratio_y = (float)y/(float)density;
      y_ = (-size/2)+ratio_y*(size);

      Cp cp;
      Eigen::Vector3f point_on_plane = Eigen::Vector3f(x_,y_,0);
      cp.point=pose*point_on_plane;
      cv::Vec3b clr;
      img->evalPixel(img->image_.rows*(1-ratio_y),img->image_.cols*ratio_x,clr);
      cp.color[0]=clr[0];
      cp.color[1]=clr[1];
      cp.color[2]=clr[2];
      cp_vector_.push_back(cp);
    }
  }
}

void Environment::generateTexturedCube(float size, Eigen::Isometry3f pose, int density){

  Eigen::Isometry3f pose_left;
  pose_left.linear()=Ry(M_PI/2);
  pose_left.translation()= Eigen::Vector3f(-size/2,0,0);
  Environment::generateTexturedPlane("images/forest.jpg", size, pose*pose_left, density);

  Eigen::Isometry3f pose_right;
  pose_right.linear()=Ry(M_PI/2);
  pose_right.translation()= Eigen::Vector3f(size/2,0,0);
  Environment::generateTexturedPlane("images/forest.jpg", size, pose*pose_right, density);

  Eigen::Isometry3f pose_up;
  pose_up.linear()=Rx(M_PI/2);
  pose_up.translation()= Eigen::Vector3f(0,size/2,0);
  Environment::generateTexturedPlane("images/folks.jpg", size, pose*pose_up, density);

  Eigen::Isometry3f pose_down;
  pose_down.linear()=Rx(M_PI/2);
  pose_down.translation()= Eigen::Vector3f(0,-size/2,0);
  Environment::generateTexturedPlane("images/folks.jpg", size, pose*pose_down, density);

  Eigen::Isometry3f pose_back;
  pose_back.linear().setIdentity();
  pose_back.translation()= Eigen::Vector3f(0,0,-size/2);
  Environment::generateTexturedPlane("images/leon.jpg", size, pose*pose_back, density);

  Eigen::Isometry3f pose_front;
  pose_front.linear().setIdentity();
  pose_front.translation()= Eigen::Vector3f(0,0,size/2);
  Environment::generateTexturedPlane("images/leon.jpg", size, pose*pose_front, density);

}

void Environment::generateCamera(std::string name, float t1, float t2, float t3, float alpha1, float alpha2, float alpha3){
  Eigen::Vector3f t_r(t1,t2,t3);
  Eigen::Isometry3f* frame_world_wrt_camera_r = new Eigen::Isometry3f;
  frame_world_wrt_camera_r->linear().setIdentity();  //TODO implement orientation
  frame_world_wrt_camera_r->translation()=t_r;
  Eigen::Isometry3f* frame_camera_wrt_world_r = new Eigen::Isometry3f;
  *frame_camera_wrt_world_r = frame_world_wrt_camera_r->inverse();
  Camera* camera = new Camera(name,lens_,aspect_,film_,resolution_,max_depth_,min_depth_,frame_camera_wrt_world_r,frame_world_wrt_camera_r);
  camera_vector_.push_back(camera);

}

bool Environment::saveEnvironment(std::string path_name, std::string dataset_name){

  const char* path_name_ = path_name.c_str(); // dataset name
  struct stat info;
  if( stat( path_name_, &info ) != 0 )
  {
    printf( "creating dataset \n" );
    std::string st = "mkdir "+path_name;
    const char *str = st.c_str();
    system(str);
  }
  else if( info.st_mode & S_IFDIR )
  {
    printf( "overwritting dataset \n" );
    std::string st = "rm -r " + path_name;
    const char *str = st.c_str();
    // std::string
    system(str);
    st = "mkdir "+path_name;
    str = st.c_str();
    system(str);
  }
  else
  {
    printf( "%s is not a directory\n", path_name );
    return 0;
  }

  std::string st = "touch "+path_name+"/"+dataset_name+".json";
  const char *str = st.c_str();
  system(str);

  json j;
  j["cameras"];

  for ( Camera* camera : camera_vector_ ){
    camera->saveRGB(path_name);
    camera->saveDepthMap(path_name);
    j["cameras"][camera->name_] = {
      {"aspect", camera->aspect_},
      {"lens", camera->lens_},
      {"resolution", camera->resolution_},
      {"width", camera->width_},
      {"max_depth", camera->max_depth_}
    };
    Eigen::Matrix3f R = camera->frame_camera_wrt_world_->linear();
    Eigen::Vector3f t = camera->frame_camera_wrt_world_->translation();

    j["cameras"][camera->name_]["frame"] = {
      R(0,0), R(0,1), R(0,2),
      R(1,0), R(1,1), R(1,2),
      R(2,0), R(2,1), R(2,2),
      t(0), t(1), t(2)
     };
     // write prettified JSON to another file
     std::ofstream o(path_name+"/"+dataset_name+".json");
     o << std::setw(4) << j << std::endl;

   }

   return 1;
}


bool Environment::loadEnvironment(std::string path_name, std::string dataset_name){

  const char* path_name_ = path_name.c_str(); // dataset name

  std::string complete_path = path_name+"/"+dataset_name+".json";

  // const char* path_name_ = complete_path.c_str(); // dataset name
  struct stat info;
  if( stat( path_name_, &info ) != 0 )
  {
    printf( "%s, dataset NOT found \n", path_name_ );
    return 0;
  }
  else if( info.st_mode & S_IFDIR )
  {
    printf( "dataset found \n" );
  }
  else
  {
    printf( "%s is not a directory\n", path_name );
    return 0;
  }

  camera_vector_.clear();
  cp_vector_.clear();



  // read a JSON file
  std::ifstream i(complete_path);
  json j;
  i >> j;

  auto cameras = j.at("cameras");
  for (json::iterator it = cameras.begin(); it != cameras.end(); ++it) {

    std::string name=it.key();

    // if(path_name+"/rgb_"+name+".png")

    // const char* path_name_ = complete_path.c_str(); // dataset name
    struct stat info_;
    std::string path_rgb_=(path_name+"/rgb_"+name+".png");
    const char* path_rgb = path_rgb_.c_str(); // dataset name
    if( stat( path_rgb, &info_ ) != 0 )
      continue;

    // struct stat info_;
    // if( stat( path_rgb, &info_ ) != 0 )
    //   continue;


    float lens;
    float aspect;
    float width;
    int resolution;
    float max_depth;
    float min_depth;
    nlohmann::basic_json<>::value_type f;
    try{
      lens = it.value().at("lens");
      max_depth = it.value().at("max_depth");
      min_depth = it.value().at("min_depth");
      aspect = it.value().at("aspect");
      width = it.value().at("width");
      resolution = it.value().at("resolution");
      f= it.value().at("frame");
    } catch (std::exception& e) {
      std::string error = ": missing values in json file";
      std::cout << error << std::endl;
      return false;
    };

    resolution_=resolution;
    film_=width;
    lens_=lens;
    max_depth_=max_depth;
    min_depth_=min_depth;
    aspect_=aspect;

    Eigen::Matrix3f R;
    R <<
      f[0], f[1], f[2],
      f[3], f[4], f[5],
      f[6], f[7], f[8];

    Eigen::Vector3f t(f[9],f[10],f[11]);
    Eigen::Isometry3f* frame_camera_wrt_world = new Eigen::Isometry3f;
    frame_camera_wrt_world->linear()=R;
    frame_camera_wrt_world->translation()=t;
    Eigen::Isometry3f* frame_world_wrt_camera = new Eigen::Isometry3f;
    *frame_world_wrt_camera=frame_camera_wrt_world->inverse();

    Camera* camera = new Camera(name,lens_,aspect_,film_,resolution_,max_depth_,
      min_depth_,frame_camera_wrt_world,frame_world_wrt_camera);

    camera->loadRGB(path_rgb);

    struct stat info__;
    std::string path_depth_=(path_name+"/depth_"+name+".png");
    const char* path_depth = path_depth_.c_str(); // dataset name
    if( stat( path_depth, &info__ ) != 0 ){
      camera_vector_.push_back(camera);
    }
    else{
      camera->loadDepthMap(path_depth_);
      camera_vector_.push_back(camera);
    }



  }


}
