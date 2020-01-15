//
// Created by liu on 19-12-5.
//

#ifndef VIO_EXAMPLE_ESTIMATOR_H
#define VIO_EXAMPLE_ESTIMATOR_H

#include <Eigen/Dense>
#include <GSLAM/core/GSLAM.h>

#include <map>
#include <vector>

class Estimator
{
public:
    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };


    Estimator(){}
    ~Estimator(){}
    virtual void ProcessIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity) = 0;
    virtual void ProcessImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header) = 0;
    virtual void ClearState() = 0;

public:
    SolverFlag _solver_flag;
    MarginalizationFlag _marginalization_flag;

    double _initial_timestamp, _td;

    std::vector<Eigen::Vector3d> _Ps;// 滑动窗口中各帧在世界坐标系下的位置
    std::vector<Eigen::Vector3d> _Vs;// 滑动窗口中各帧在世界坐标系下的速度
    std::vector<Eigen::Matrix3d> _Rs;// 滑动窗口中各帧在世界坐标系下的旋转
    std::vector<Eigen::Vector3d> _Bas;// 滑动窗口中各帧对应的加速度偏置
    std::vector<Eigen::Vector3d> _Bgs;// 滑动窗口中各帧对应的陀螺仪偏置

    std::vector<double> _headers;
};


#endif //VIO_EXAMPLE_ESTIMATOR_H
