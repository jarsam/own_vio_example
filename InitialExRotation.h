//
// Created by liu on 19-12-10.
//

#ifndef VIO_EXAMPLE_INITIALEXROTATION_H
#define VIO_EXAMPLE_INITIALEXROTATION_H

#include <vector>

#include "Parameters.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class InitialExRotation
{
public:
    InitialExRotation();
    bool CalibrationExRotation(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres,
                               Eigen::Quaterniond &delta_q_imu, Eigen::Matrix3d &calib_ric_result);

private:
    Eigen::Matrix3d SolveRelativeR(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres);
    void DecomposeE(cv::Mat E, cv::Mat_<double> &R1, cv::Mat_<double> &R2, cv::Mat_<double> &t1, cv::Mat_<double> &t2);
    double TestTriangulation(const std::vector<cv::Point2f> &l, const std::vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double > t);

private:
    int _frame_count;
    std::vector<Eigen::Matrix3d> _Rc;
    std::vector<Eigen::Matrix3d> _Rimu;
    std::vector<Eigen::Matrix3d> _Rc_g;
    Eigen::Matrix3d _ric;
};


#endif //VIO_EXAMPLE_INITIALEXROTATION_H
