//
// Created by liu on 19-12-10.
//

#ifndef VIO_EXAMPLE_INITIALALIGNMENT_H
#define VIO_EXAMPLE_INITIALALIGNMENT_H

#include <iostream>
#include <map>
#include <vector>
#include <memory>

#include <Eigen/Dense>

#include "IntegrationBase.h"

class ImageFrame
{
public:
    ImageFrame(){}
    ImageFrame(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &points, double t)
        :_t(t), _points(points){}

    // feature_id, feature组合, camera_id, feature信息
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> _points;
    double _t;
    // Imu到第l帧相机的R
    Eigen::Matrix3d _R;
    // 第l帧相机到当前帧相机的位移,只不过没有尺度信息
    // 尺度是初始化的时候的第l帧到初始化帧的尺度
    Eigen::Vector3d _T;
    IntegrationBase* _pre_integration;
    bool _keyframe_flag;
};

bool VisualImuAlignment(std::map<double, ImageFrame> &all_image_frame, std::vector<Eigen::Vector3d> &bgs, Eigen::Vector3d &g, Eigen::VectorXd &x);


#endif //VIO_EXAMPLE_INITIALALIGNMENT_H
