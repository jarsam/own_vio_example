//
// Created by liu on 19-12-20.
//

#ifndef VIO_EXAMPLE_PROJECTIONFACTOR_H
#define VIO_EXAMPLE_PROJECTIONFACTOR_H

#include <ceres/ceres.h>
#include <Eigen/Dense>

#include "Utility.h"
#include "Parameters.h"

class ProjectionFactor: public ceres::SizedCostFunction<2, 7, 7, 7, 1>
{
public:
    ProjectionFactor(const Eigen::Vector3d &pts_i, const Eigen::Vector3d &pts_j):
        _pts_i(pts_i), _pts_j(pts_j)
    {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d    Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d    Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        Eigen::Vector3d    tic(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        double inv_dep_i = parameters[3][0]; // i时刻相机坐标系下的map point的逆深度
        // 将第i frame下的3D点转到第j frame坐标系下
        Eigen::Vector3d pts_camera_i = _pts_i / inv_dep_i;                 // pt in ith camera frame, 归一化平面
        Eigen::Vector3d pts_imu_i    = qic * pts_camera_i + tic;          // pt in ith body frame
        Eigen::Vector3d pts_w        = Qi * pts_imu_i + Pi;               // pt in world frame
        Eigen::Vector3d pts_imu_j    = Qj.inverse() * (pts_w - Pj);       // pt in jth body frame
        Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic); // pt in jth camera frame

        Eigen::Map<Eigen::Vector2d> residual(residuals);
        double dep_j = pts_camera_j.z();
        residual = (pts_camera_j / dep_j).head<2>() - _pts_j.head<2>();
        residual = _sqrt_info * residual; // 转成 与量纲无关的马氏距离
        if (jacobians)
        {
            Eigen::Matrix3d Ri  = Qi.toRotationMatrix();
            Eigen::Matrix3d Rj  = Qj.toRotationMatrix();
            Eigen::Matrix3d ric = qic.toRotationMatrix();

            Eigen::Matrix<double, 2, 3> reduce(2, 3);
            reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
                0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
            reduce = _sqrt_info * reduce;
            // 因为残差是2， 四个参数块对应的雅克比矩阵分别是：2*7， 2*7, 2*7， 2*1
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

                Eigen::Matrix<double, 3, 6> jaco_i;
                jaco_i.leftCols<3>()  = ric.transpose() * Rj.transpose();
                jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::SkewSymmetric(pts_imu_i);

                jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
                jacobian_pose_i.rightCols<1>().setZero();
            }

            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

                Eigen::Matrix<double, 3, 6> jaco_j;
                jaco_j.leftCols<3>()  = ric.transpose() * -Rj.transpose();
                jaco_j.rightCols<3>() = ric.transpose() * Utility::SkewSymmetric(pts_imu_j);

                jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
                jacobian_pose_j.rightCols<1>().setZero();
            }

            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);

                Eigen::Matrix<double, 3, 6> jaco_ex;
                jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
                Eigen::Matrix3d tmp_r = ric.transpose() *  Rj.transpose() * Ri * ric;
                jaco_ex.rightCols<3>() =
                    -tmp_r * Utility::SkewSymmetric(pts_camera_i) +
                    Utility::SkewSymmetric(tmp_r * pts_camera_i) +
                    Utility::SkewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));

                jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
                jacobian_ex_pose.rightCols<1>().setZero();
            }

            if (jacobians[3])
            {
                Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
                jacobian_feature =
                    reduce * ric.transpose() * Rj.transpose() * Ri * ric * _pts_i * -1.0 / (inv_dep_i * inv_dep_i);
            }
        }
        return true;
    }

    void check(double **parameters);

    Eigen::Vector3d _pts_i, _pts_j;
    Eigen::Matrix<double, 2, 3> _tangent_base;
    Eigen::Matrix2d _sqrt_info =
        (para._camera_intrinsics[0] + para._camera_intrinsics[1]) / 1.5 * 2.0 * Eigen::Matrix2d::Identity();
};

#endif //VIO_EXAMPLE_PROJECTIONFACTOR_H
