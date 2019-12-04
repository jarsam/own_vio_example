#pragma once

#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <GSLAM/core/GSLAM.h>

#include "FeatureTracker.h"

struct ImuMessage
{
    double _header;
    Eigen::Vector3d _linear_acceleration;
    Eigen::Vector3d _angular_velocity;
};
typedef std::shared_ptr<ImuMessage> ImuMessagePtr;

struct ImageMessage
{
    double _header;
    std::vector<Eigen::Vector3d> _points;
    std::vector<int> _points_id;
    std::vector<double> _point_u;
    std::vector<double> _point_v;
    std::vector<double> _point_x_velocity;
    std::vector<double> _point_y_velocity;
};
typedef std::shared_ptr<ImageMessage> ImageMessagePtr;

class System
{
public:
    System(std::string data_path): _data_path(data_path){
        ReadParameters();
        _pub_count = 0;
        _tracker_data.resize(svar.GetInt("number_of_camera", 1));
    }

    bool PubImageData();
    bool PubImuData();
    void ProcessBackEnd();
    void Draw();

private:
    void GetImageData(double stamp_sec, cv::Mat &img);
    void GetImuData(double stamp_sec, const Eigen::Vector3d &gyr, const Eigen::Vector3d &acc);

    void ReadParameters();

private:
    std::string _data_path;
    const double _delay_times = 2.0;

    bool _init_feature = 0;
    bool _init_imu = 0;
    bool _first_image_flag = 1;

    double _pub_count;
    double _first_image_time;
    double _last_image_time;

    std::vector<FeatureTracker> _tracker_data;
};