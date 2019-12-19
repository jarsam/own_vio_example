//
// Created by liu on 19-12-19.
//

#ifndef VIO_EXAMPLE_PROJECTIONTDFACTOR_H
#define VIO_EXAMPLE_PROJECTIONTDFACTOR_H

#include <ceres/ceres.h>
#include <Eigen/Dense>

#include "Utility.h"
#include "Parameters.h"

class ProjectionTdFactor: public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1>
{
public:
    ProjectionTdFactor(const Eigen::Vector3d &pts_i, const Eigen::Vector3d &pts_j, const Eigen::Vector2d &velocity_i,
                       const Eigen::Vector2d &velocity_j, const double td_i, const double td_j, const double row_i,
                       const double row_j)
                       : _pts_i(pts_i), _pts_j(pts_j), _td_i(td_i), _td_j(td_j)
    {
        _velocity_i.x() = velocity_i.x();
        _velocity_i.y() = velocity_i.y();
        _velocity_i.z() = 0;
        _velocity_j.x() = velocity_j.x();
        _velocity_j.y() = velocity_j.y();
        _velocity_j.z() = 0;

        _row_i = row_i - para._camera_intrinsics[3];
        _row_j = row_j - para._camera_intrinsics[3];
    }

    /**
     * 添加对 imu-camera 时间戳不完全同步和 Rolling shutter 相机的支持
     * 主要的思路就是通过前端光流计算得到每个角点在归一化的速度，
     * 根据 imu-camera时间戳的时间同步误差和Rolling shutter相机做一次rolling的时间，对角点的归一化坐标进行调整
     *
     * pts_i 是角点在归一化平面的坐标
     * td    表示imu-camera时间戳的时间同步误差，是待优化项
     * TR    表示Rolling shutter相机做一次rolling的时间
     * row_i 是角点图像坐标的纵坐标
     * ROW   图像坐标纵坐标的最大值
     * velocity_i 是该角点在归一化平面的运动速度
     *
     * 因为在处理imu数据的时候，已经减过一次时间同步误差，因此修正后的时间误差是td - td_i
     * TR / ROW * row_i 是相机 rolling 到这一行时所用的时间
     * 最后得到的pts_i_td是处理时间同步误差和Rolling shutter时间后，角点在归一化平面的坐标
     */
     
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d    Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        Eigen::Vector3d    Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d    tic(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        double inv_dep_i = parameters[3][0];
    }

    static Eigen::Matrix2d _sqrt_info;

    Eigen::Vector3d _pts_i, _pts_j;
    Eigen::Vector3d _velocity_i, _velocity_j;
    Eigen::Matrix<double, 2, 3> _tangent_base;

    double _td_i, _td_j;
    double _row_i, _row_j;
    static double _sum_t;
};

#endif //VIO_EXAMPLE_PROJECTIONTDFACTOR_H
