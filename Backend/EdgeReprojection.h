//
// Created by liu on 20-1-7.
//

#ifndef VIO_EXAMPLE_EDGEREPROJECTION_H
#define VIO_EXAMPLE_EDGEREPROJECTION_H

#include <memory>
#include <string>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include "EigenTypes.h"
#include "Edge.h"
#include "Utility.h"

/*
 * 此边为视觉重投影误差, 为三元边, 与之相连的顶点有:
 * 路标点的逆深度, 第一次观测到该路标点的source camera的位姿T_world_from_body1(T1)
 * 和观测到该路标点的measurement camera位姿T_world_from_body2(T1)
 * 顺序必须为逆深度, T1, T2
 */
class EdgeReprojection: public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // FIXME: 为什么是4个节点
    // 这里算上了ex_pose
    EdgeReprojection(const Vec3 &pts_i, const Vec3 &pts_j):
        Edge(2, 4, std::vector<std::string>{"VertexInverseDepth", "VertexPose", "VertexPose", "VertexPose"}){
        _pts_i = pts_i;
        _pts_j = pts_j;
    }

    virtual std::string TypeInfo() const override {return "EdgeReprojection";}

    virtual void ComputeResidual() override{
        double inv_dep_i = _verticies[0]->Parameters()[0];

        VecX param_i = _verticies[1]->Parameters();
        Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
        Vec3 Pi = param_i.head<3>();

        VecX param_j = _verticies[2]->Parameters();
        Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
        Vec3 Pj = param_j.head<3>();

        VecX param_ext = _verticies[3]->Parameters();
        Qd qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
        Vec3 tic = param_ext.head<3>();

        Vec3 pts_camera_i = _pts_i / inv_dep_i;
        Vec3 pts_imu_i = qic * pts_camera_i + tic;
        Vec3 pts_w = Qi * pts_imu_i + Pi;
        Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);
        Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);

        double dep_j = pts_camera_j.z();
        _residual = (pts_camera_j / dep_j).head<2>() - _pts_j.head<2>(); // J^t * J * delta_x = -J^t * r
    }

    virtual void ComputeJacobians() override{
        double inv_dep_i = _verticies[0]->Parameters()[0];

        VecX param_i = _verticies[1]->Parameters();
        Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
        Vec3 Pi = param_i.head<3>();

        VecX param_j = _verticies[2]->Parameters();
        Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
        Vec3 Pj = param_j.head<3>();

        VecX param_ext = _verticies[3]->Parameters();
        Qd qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
        Vec3 tic = param_ext.head<3>();

        Vec3 pts_camera_i = _pts_i / inv_dep_i;
        Vec3 pts_imu_i = qic * pts_camera_i + tic;
        Vec3 pts_w = Qi * pts_imu_i + Pi;
        Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);
        Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);

        double dep_j = pts_camera_j.z();

        Mat33 Ri = Qi.toRotationMatrix();
        Mat33 Rj = Qj.toRotationMatrix();
        Mat33 ric = qic.toRotationMatrix();
        Mat23 reduce(2, 3);
        reduce << 1./dep_j, 0, -pts_camera_j(0)/(dep_j * dep_j), 0, 1./dep_j, -pts_camera_j(1)/(dep_j * dep_j);

        Eigen::Matrix<double, 2, 6> jacobian_pose_i;
        Eigen::Matrix<double, 3, 6> jaco_i;
        jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
        jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * - Sophus::SO3d::hat(pts_imu_i);
        jacobian_pose_i.leftCols<6>() = reduce * jaco_i;

        Eigen::Matrix<double, 2, 6> jacobian_pose_j;
        Eigen::Matrix<double, 3, 6> jaco_j;
        jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
        jaco_j.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_j);
        jacobian_pose_j.leftCols<6>() = reduce * jaco_j;

        Eigen::Vector2d jacobian_feature;
        jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * _pts_i * -1.0 / (inv_dep_i * inv_dep_i);

        Eigen::Matrix<double, 2, 6> jacobian_ex_pose;
        Eigen::Matrix<double, 3, 6> jaco_ex;
        jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
        Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
        jaco_ex.rightCols<3>() = -tmp_r * Utility::SkewSymmetric(pts_camera_i) + Utility::SkewSymmetric(tmp_r * pts_camera_i) +
                                 Utility::SkewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
        jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;

        _jacobians[0] = jacobian_feature;
        _jacobians[1] = jacobian_pose_i;
        _jacobians[2] = jacobian_pose_j;
        _jacobians[3] = jacobian_ex_pose;
    }

private:
    Vec3 _pts_i, _pts_j;
};

#endif //VIO_EXAMPLE_EDGEREPROJECTION_H
