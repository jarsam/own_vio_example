//
// Created by liu on 19-12-19.
//

#ifndef VIO_EXAMPLE_PROJECTIONTDFACTOR_H
#define VIO_EXAMPLE_PROJECTIONTDFACTOR_H

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <GSLAM/core/GSLAM.h>

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

        _row_i = row_i - para._height / 2;
        _row_j = row_j - para._height / 2;
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d    Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        Eigen::Vector3d    Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d    tic(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        double inv_dep_i = parameters[3][0];
        double td = parameters[4][0];

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

        Eigen::Vector3d pts_i_td, pts_j_td;
        double TR = svar.GetDouble("tr", 0);
        double ROW = para._height;
        pts_i_td = _pts_i - (td - _td_i + TR / ROW * _row_i) * _velocity_i;
        pts_j_td = _pts_j - (td - _td_j + TR / ROW * _row_j) * _velocity_j;

        // 将第i帧的3D点转到第j帧坐标系下
        Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
        Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
        Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
        Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
        Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
        Eigen::Map<Eigen::Vector2d> residual(residuals);

        double dep_j = pts_camera_j.z();
        // 对比归一化平面上的residual
        residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
        residual = _sqrt_info * residual;

        // 雅克比矩阵为视觉误差对两个时刻的状态量, 外参以及逆深度求导.
        if (jacobians){
            Eigen::Matrix3d Ri = Qi.toRotationMatrix();
            Eigen::Matrix3d Rj = Qj.toRotationMatrix();
            Eigen::Matrix3d ric = qic.toRotationMatrix();
            Eigen::Matrix<double, 2, 3> reduce(2, 3);

            // 链式求导中视觉误差对第j帧归一化点的求导
            reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
                0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);

            // 链式求导中第j帧归一化点对两个时刻的状态量, 外参以及逆深度求导.
            // jacobians[0]是第j帧归一化点对i时刻的状态量求偏导
            // 要注意的是这里是对世界坐标系下的第i帧做偏导
            if (jacobians[0]){
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > jacobian_pose_i(jacobians[0]);
                Eigen::Matrix<double, 3, 6> jaco_i;
                jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
                jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::SkewSymmetric(pts_imu_i);

                jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
                jacobian_pose_i.rightCols<1>().setZero();
            }
            // 这里是对j时刻的状态量求偏导
            // 注意转置矩阵求偏导需要左乘(I-δΘ), 而非转置则是右乘(I-δΘ)
            if (jacobians[1]){
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > jacobian_pose_j(jacobians[1]);
                Eigen::Matrix<double, 3, 6> jaco_j;
                jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
                jaco_j.leftCols<3>() = ric.transpose() * Utility::SkewSymmetric(pts_imu_j);

                jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
                jacobian_pose_j.rightCols<1>().setZero();
            }
            // 对外参进行求导
            if (jacobians[2]){
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > jacobian_ex_pose(jacobians[2]);
                Eigen::Matrix<double, 3, 6> jaco_ex;
                jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
                Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
                jaco_ex.rightCols<3>() = -tmp_r * Utility::SkewSymmetric(pts_camera_i) + Utility::SkewSymmetric(tmp_r * pts_camera_i) +
                                         Utility::SkewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
                jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
                jacobian_ex_pose.rightCols<1>().setZero();
            }
            // 对特征点逆深度求偏导
            if (jacobians[3]){
                Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
                jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
            }
            // 对TD求导
            if (jacobians[4]){
                Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[4]);
                jacobian_td = reduce * ric.transpose() * Rj.transpose() * Ri * ric * _velocity_i / inv_dep_i * -1.0 +
                    _sqrt_info * _velocity_j.head(2);
            }
        }

        return true;
    }

    Eigen::Matrix2d _sqrt_info =
        (para._camera_intrinsics[0] + para._camera_intrinsics[1]) / 1.5 * 2.0 * Eigen::Matrix2d::Identity();

    Eigen::Vector3d _pts_i, _pts_j;
    Eigen::Vector3d _velocity_i, _velocity_j;
    Eigen::Matrix<double, 2, 3> _tangent_base;

    double _td_i, _td_j;
    double _row_i, _row_j;
    static double _sum_t;
};

#endif //VIO_EXAMPLE_PROJECTIONTDFACTOR_H
