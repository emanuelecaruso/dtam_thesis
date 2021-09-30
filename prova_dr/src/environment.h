#pragma once
#include "camera.h"

class Environment{
  public:
    // cameras parameters
    int resolution_;
    float film_;
    float lens_;
    float aspect_;
    float min_depth_;
    float max_depth_;
    // environment state
    cpVector cp_vector_;

    CameraVector camera_vector_; // vector containing pointers to camera objects

    Environment(int resolution = 640, float aspect = 1, float film = 0.024,
                            float lens = 0.035, float max_depth =4.2)
        {
          resolution_=resolution; film_=film; lens_=lens;
          aspect_=aspect; max_depth_=max_depth;
        }

    void generateSinusoidalSurface(float picks_depth, int density);
    void generateTexturedPlane(std::string path, float size, Eigen::Isometry3f pose, int density);
    void generateTexturedCube(float size, Eigen::Isometry3f pose, int density);
    void generateLine(int density);

    void generateCamera(std::string name, float t1, float t2, float t3, float alpha1, float alpha2, float alpha3);

    bool saveEnvironment(std::string path_name, std::string dataset_name);
    bool loadEnvironment(std::string path_name, std::string dataset_name);
};
