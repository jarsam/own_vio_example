//
// Created by liu on 19-12-10.
//

#ifndef VIO_EXAMPLE_INITIALEXROTATION_H
#define VIO_EXAMPLE_INITIALEXROTATION_H

#include <vector>

#include "Parameters.h"
#include "Utility.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <GSLAM/core/GSLAM.h>

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
    // _Rc代表的是相机的旋转
    std::vector<Eigen::Matrix3d> _Rc;
    // _Rimu代表的是Imu的旋转
    std::vector<Eigen::Matrix3d> _Rimu;
    // 由_ric和Imu信息算出的两帧之间旋转.
    std::vector<Eigen::Matrix3d> _Rc_g;
    // _ric 是计算出来的相机到Imu的旋转.
    Eigen::Matrix3d _ric;
};


#endif //VIO_EXAMPLE_INITIALEXROTATION_H
