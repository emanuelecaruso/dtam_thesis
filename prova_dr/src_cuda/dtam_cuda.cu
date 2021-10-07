#include "dtam_cuda.cuh"
#include <math.h>
#include "utils.h"
#include <stdlib.h>
#include "defs.h"
#include "cuda_utils.cuh"


void Dtam::addCamera(Camera_cpu* camera_cpu, Camera_gpu* camera_gpu){
  camera_vector_cpu_.push_back(camera_cpu);
  camera_vector_gpu_.push_back(camera_gpu);
  mapper_->camera_vector_cpu_.push_back(camera_cpu);
  mapper_->camera_vector_gpu_.push_back(camera_gpu);
  tracker_->camera_vector_cpu_.push_back(camera_cpu);
  tracker_->camera_vector_gpu_.push_back(camera_gpu);
}

bool Dtam::setReferenceCamera(int index_r){

  int num_cameras = camera_vector_cpu_.size();

  if (index_r<0 || index_r>=num_cameras)
    return false;

  index_r_ = index_r;
  mapper_->index_r_ = index_r;
  tracker_->index_r_ = index_r;

  mapper_->Initialize();

  return true;

}

void Dtam::showImgs(int scale){
  int resolution=camera_vector_cpu_[index_r_]->resolution_;

  cv::Mat_< float > q_gradient;
  (mapper_->gradient_q).download(q_gradient);
  cv::Mat_< float > resized_image_q_gradient;
  cv::resize(q_gradient, resized_image_q_gradient, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  cv::imshow("gradient_q", resized_image_q_gradient);

  cv::Mat_< float > ad_comp;


  cv::Mat_< float > d_1;
  mapper_->d.download(d_1);
  cv::Mat_< float > resized_image_d_1;
  cv::resize(d_1, resized_image_d_1, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  cv::Mat_< float > a_1;
  mapper_->a.download(a_1);
  cv::Mat_< float > resized_image_a;
  cv::resize(a_1, resized_image_a, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  cv::hconcat(resized_image_d_1,resized_image_a,ad_comp);

  cv::Mat_< float > gt_gterr_comp;

  cv::Mat_< float > gt;
  mapper_->depth_groundtruth_.download(gt);
  cv::Mat_< float > resized_image_gt;
  cv::resize(gt, resized_image_gt, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  cv::Mat_< float > gterr;
  cv::absdiff(resized_image_gt, resized_image_d_1, gterr);
  // cv::absdiff(resized_image_gt, resized_image_pa_1, gterr);

  cv::hconcat(resized_image_gt,gterr,gt_gterr_comp);

  cv::Mat_< float > out;
  cv::vconcat(ad_comp,gt_gterr_comp,out);

  cv::imshow("comparison", out);


  cv::Mat_< float > state;
  (camera_vector_cpu_[index_r_]->invdepth_map_gpu_).download(state);
  cv::Mat_< float > resized_image_state;
  cv::resize(state, resized_image_state, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  cv::imshow("state", resized_image_state);

  // camera_vector_cpu_[index_r_]->invdepth_map_->show(800/camera_vector_cpu_[index_r_]->resolution_);

  // Image< float >* invdepth_groundtruth = new Image< float >("invdepth groundtruth");
  // mapper_->depth_groundtruth_.download(invdepth_groundtruth->image_);
  // invdepth_groundtruth->show(scale/camera_vector_cpu_[index_m]->resolution_);

  // cv::Mat_< float > d_gradient;
  // (mapper_->gradient_d).download(d_gradient);
  // cv::Mat_< float > resized_image_d_gradient;
  // cv::resize(d_gradient, resized_image_d_gradient, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  // cv::imshow("gradient_d", resized_image_d_gradient);

  // cv::Mat_< float > q_1;
  // mapper_->q.download(q_1);
  // cv::Mat_< float > resized_image_q_1;
  // cv::resize(q_1, resized_image_q_1, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  // cv::imshow("q_1", resized_image_q_1);
  //


  cv::waitKey(0);

}

void Dtam::test_mapping(Environment_gpu* environment){

  double t_start;  // time start for computing computation time
  double waitKeyDelay=0;

  bool frame_available=true;
  bool set_reference=true;
  bool init=true;
  int it=0;
  int frames_computed_=0;

  mapper_->depthSampling(environment);

  t_start=getTime();
  while (true){

    // considering 30 fps camera
    float fps=30;
    int current_frame=int((getTime()-t_start-waitKeyDelay)/((1.0/fps)*1000));
    int frames_delta=current_frame-frames_computed_;
    if(frames_delta>=0){
      frame_available=true;
      if(frames_computed_>=environment->camera_vector_cpu_.size()){
        break;
      }
      Camera_cpu* camera_cpu=environment->camera_vector_cpu_[frames_computed_];
      Camera_gpu* camera_gpu=environment->camera_vector_gpu_[frames_computed_];
      // set groundtruth pose
      camera_cpu->setGroundtruthPose(camera_gpu);
      // load camera
      addCamera(camera_cpu,camera_gpu);
      frames_computed_+=(frames_delta+1);
      if (frames_delta>0)
        std::cout << frames_delta+1 << " frames has been skipped!" << std::endl;
      std::cout << "\nFrame n: " << frames_computed_-1 << std::endl;
    }

    if (frame_available){

      if (set_reference){
        setReferenceCamera(frames_computed_-1);
        set_reference=false;
      }
      else{
        int index_m=frames_computed_-1;
        // printf("%i\n", index_m);

        mapper_->UpdateCostVolume(index_m,(frames_computed_-1)<2 );

        if(index_m==(index_r_+1)){
          mapper_->ComputeCostVolumeMin();

          init = false;
        }
        else if(index_m>(index_r_+2)){
          mapper_->UpdateState();
        }

      }


      frame_available=false;
    }

    else if(!init){

      double t_s1=getTime();
      // float cr=0.484; float rr=0.465;  //occlusion
      float cr=0.51; float rr=0.98;  //strange down
      // float cr=0.61; float rr=0.53;  //hightex1
      // float cr=0.61; float rr=0.53;  //hightex2
      // float cr=0.95; float rr=0.87;  //corner dr
      // float cr=0.95; float rr=0.08;  //corner ur
      // float cr=0.5; float rr=0.9;  //hightex cube

      int index_m=frames_computed_-1;
      int col=cr*camera_vector_cpu_[0]->resolution_;
      int row=rr*camera_vector_cpu_[0]->resolution_/camera_vector_cpu_[0]->aspect_;
      // mapper_->StudyCostVolumeMin(index_m, row, col, true);
      double t_e1=getTime();
      double delta1=t_e1-t_s1;
      waitKeyDelay+=delta1;

      mapper_->Regularize();

      double t_s2=getTime();
      showImgs(640);
      double t_e2=getTime();
      double delta2=t_e2-t_s2;
      waitKeyDelay+=delta2;

    }


  }

}


void Dtam::test_tracking(Environment_gpu* environment){

  double t_start;  // time start for computing computation time
  double waitKeyDelay=0;

  bool frame_available=true;
  bool set_reference=true;
  bool init=true;
  int it=0;
  int frames_computed_=0;

  t_start=getTime();
  while (true){

    // considering 30 fps camera
    float fps=30;
    int current_frame=int((getTime()-t_start-waitKeyDelay)/((1.0/fps)*1000));
    int frames_delta=current_frame-frames_computed_;
    if(frames_delta>=0){
      frame_available=true;
      if(frames_computed_>=environment->camera_vector_cpu_.size()){
        break;
      }
      Camera_cpu* camera_cpu=environment->camera_vector_cpu_[frames_computed_];
      Camera_gpu* camera_gpu=environment->camera_vector_gpu_[frames_computed_];
      // load camera
      addCamera(camera_cpu,camera_gpu);


      frames_computed_+=(frames_delta+1);
      if (frames_delta>0)
        std::cout << frames_delta+1 << " frames has been skipped!" << std::endl;
      std::cout << "\nFrame n: " << frames_computed_-1 << std::endl;
    }

    if (frame_available){

      if (set_reference){
        setReferenceCamera(frames_computed_-1);
        set_reference=false;
      }
      else{
        int index_m=frames_computed_-1;
        tracker_->printPoseComparison(index_m);

      }



      frame_available=false;
    }


  }

}
