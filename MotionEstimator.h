//
// Created by liu on 19-12-11.
//

#ifndef VIO_EXAMPLE_MOTIONESTIMATOR_H
#define VIO_EXAMPLE_MOTIONESTIMATOR_H

#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class MotionEstimator
{
public:
    bool SolveRelativeRT(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres, Eigen::Matrix3d &R, Eigen::Vector3d &T);
};


#endif //VIO_EXAMPLE_MOTIONESTIMATOR_H
