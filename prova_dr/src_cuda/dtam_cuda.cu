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
}

void Dtam::showImgs(int scale){
  int resolution=camera_vector_cpu_[index_r_]->resolution_;

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

  cv::hconcat(resized_image_gt,gterr,gt_gterr_comp);

  cv::Mat_< float > out;
  cv::vconcat(ad_comp,gt_gterr_comp,out);

  cv::imshow("comparison", out);

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
  cv::Mat_< float > q_gradient;
  (mapper_->gradient_q).download(q_gradient);
  cv::Mat_< float > resized_image_q_gradient;
  cv::resize(q_gradient, resized_image_q_gradient, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  cv::imshow("gradient_q", resized_image_q_gradient);

  cv::Mat_< float > pa_1;
  mapper_->points_added_.download(pa_1);
  cv::Mat_< float > resized_image_pa_1;
  cv::resize(pa_1, resized_image_pa_1, cv::Size(), scale/resolution, scale/resolution, cv::INTER_NEAREST );
  cv::imshow("pa 1", resized_image_pa_1);

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
      // load camera (already with pose)
      addCamera(environment->camera_vector_cpu_[frames_computed_],environment->camera_vector_gpu_[frames_computed_]);
      frames_computed_+=(frames_delta+1);
      if (frames_delta>0)
        std::cout << frames_delta+1 << " frames has been skipped!" << std::endl;
      std::cout << "\nFrame n: " << frames_computed_-1 << std::endl;
    }

    if (frame_available){

      if (set_reference){
        mapper_->setReferenceCamera(frames_computed_-1);
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
      double t_s1=getTime();
      mapper_->StudyCostVolumeMin(index_m, row, col, true);
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

  // double t_start;  // time start for computing computation time
  // double waitKeyDelay=0;
  //
  // bool frame_available=true;
  // bool set_reference=true;
  // bool init=true;
  // int it=0;
  // frames_computed_=0;
  //
  // depthSampling(environment);
  //
  // t_start=getTime();
  // while (true){
  //
  //   // considering 30 fps camera
  //   float fps=30;
  //   int current_frame=int((getTime()-t_start-waitKeyDelay)/((1.0/fps)*1000));
  //   int frames_delta=current_frame-frames_computed_;
  //   if(frames_delta>=0){
  //     frame_available=true;
  //     if(frames_computed_>=environment->camera_vector_cpu_.size()){
  //       break;
  //     }
  //     // load camera (already with pose)
  //     Dtam::addCamera(environment->camera_vector_cpu_[frames_computed_],environment->camera_vector_gpu_[frames_computed_]);
  //     frames_computed_+=(frames_delta+1);
  //     if (frames_delta>0)
  //       std::cout << frames_delta+1 << " frames has been skipped!" << std::endl;
  //     std::cout << "\nFrame n: " << frames_computed_-1 << std::endl;
  //   }
  //
  //   if (frame_available){
  //
  //     if (set_reference){
  //       Dtam::setReferenceCamera(frames_computed_-1);
  //       set_reference=false;
  //     }
  //     else{
  //       int index_m=frames_computed_-1;
  //       // printf("%i\n", index_m);
  //       Dtam::prepareCameraForDtam(index_m);
  //       Dtam::UpdateCostVolume(index_m);
  //
  //       if(index_m==(index_r_+1)){
  //         Dtam::ComputeCostVolumeMin();
  //
  //         d = camera_vector_cpu_[index_r_]->invdepth_map_gpu_.clone();
  //         a = camera_vector_cpu_[index_r_]->invdepth_map_gpu_.clone();
  //         q.create(d.rows*2,d.cols*2,CV_32FC1);
  //         gradient_d.create(d.rows*2,d.cols*2,CV_32FC1);
  //         gradient_q.create(d.rows,d.cols,CV_32FC1);
  //
  //
  //         init = false;
  //       }
  //       else if(index_m>(index_r_+2)){
  //         UpdateState();
  //       }
  //
  //     }
  //
  //
  //     frame_available=false;
  //   }
  //   else if(!init){
  //
  //     // float cr=0.484; float rr=0.465;  //occlusion
  //     float cr=0.51; float rr=0.98;  //strange down
  //     // float cr=0.61; float rr=0.53;  //hightex1
  //     // float cr=0.61; float rr=0.53;  //hightex2
  //     // float cr=0.95; float rr=0.87;  //corner dr
  //     // float cr=0.95; float rr=0.08;  //corner ur
  //     // float cr=0.5; float rr=0.9;  //hightex cube
  //
  //     int index_m=frames_computed_-1;
  //     int col=cr*camera_vector_cpu_[0]->resolution_;
  //     int row=rr*camera_vector_cpu_[0]->resolution_/camera_vector_cpu_[0]->aspect_;
  //     // Dtam::StudyCostVolumeMin(index_m, camera_data_for_dtam_, row, col, true);
  //
  //     if(theta_>theta_end_){
  //     //   // if(count_>=3){
  //         Dtam::Regularize();
  //     //   // }
  //     }
  //     double t_s=getTime();
  //     Dtam::showImgs(640);
  //     cv::waitKey(0);
  //     double t_e=getTime();
  //     double delta=t_e-t_s;
  //     waitKeyDelay+=delta;
  //
  //   }
  //
  //
  // }

}
