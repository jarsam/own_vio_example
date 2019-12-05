#pragma once

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <GSLAM/core/GSLAM.h>

#include "FeatureTracker.h"
#include "Estimator.h"

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
    // _points 包含归一化且去畸变的点.
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
    System(std::string data_path): _data_path(data_path), _start_backend(true), _pub_count(0), _imu_current_time(-1){
        ReadParameters();
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
    std::vector<std::pair<std::vector<ImuMessagePtr>, ImageMessagePtr>> GetMeasurements();

private:
    std::string _data_path;
    const double _delay_times = 2.0;

    bool _init_pub = false;
    bool _init_feature = 0;
    bool _init_imu = 0;
    bool _first_image_flag = 1;
    bool _start_backend;

    // pub的帧数
    double _pub_count;
    double _first_image_time;
    double _last_image_time;

    double _last_imu_time;
    // imu中的时间
    double _imu_current_time;

    //
    std::mutex _feature_buf_mutex;
    std::mutex _estimator_mutex;

    std::queue<ImageMessagePtr> _feature_buf;
    std::queue<ImuMessagePtr> _imu_buf;

    std::condition_variable _con;

    std::vector<FeatureTracker> _tracker_data;

    Estimator _estimator;
};