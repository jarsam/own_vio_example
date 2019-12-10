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

    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> _points;
    double _t;
    Eigen::Matrix3d _R;
    Eigen::Vector3d _T;
    std::shared_ptr<IntegrationBase> _pre_integration;
    bool _keyframe_flag;
};

bool VisualImuAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &bgs, Eigen::Vector3d &g, Eigen::Vector3d &x);


#endif //VIO_EXAMPLE_INITIALALIGNMENT_H
