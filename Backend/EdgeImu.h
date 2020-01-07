//
// Created by liu on 20-1-7.
//

#ifndef VIO_EXAMPLE_EDGEIMU_H
#define VIO_EXAMPLE_EDGEIMU_H

#include <sophus/se3.hpp>

#include "Edge.h"
#include "IntegrationBase.h"

class EdgeImu: public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    explicit EdgeImu(IntegrationBase* pre_integration): _pre_integration(pre_integration),
        Edge(15, 4, std::vector<std::string>{"VertexPose", "VertexSpeedBias", "VertexPose", "VertexSpeedBias"}){}

    virtual std::string TypeInfo() const override {return "EdgeImu";}

    virtual void ComputeResidual() override{
        VecX param_0 = _verticies[0]->Parameters();
        Qd Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
        Vec3 Pi = param_0.head<3>();

        VecX param_1 = _verticies[1]->Parameters();
        Vec3 Vi = param_1.head<3>();
        Vec3 Bai = param_1.segment(3, 3);
        Vec3 Bgi = param_1.tail<3>();

        VecX param_2 = _verticies[2]->Parameters();
        Qd Qj(param_2[6], param_2[3], param_2[4], param_2[5]);
        Vec3 Pj = param_2.head<3>();

        VecX param_3 = _verticies[3]->Parameters();
        Vec3 Vj = param_3.head<3>();
        Vec3 Baj = param_3.segment(3, 3);
        Vec3 Bgj = param_3.tail<3>();

        _residual = _pre_integration->Evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);
        SetInformation(_pre_integration->_covariance.inverse());
    }

    virtual void ComputeJacobians() override{
        VecX param_0 = _verticies[0]->Parameters();
        Qd Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
        Vec3 Pi = param_0.head<3>();

        VecX param_1 = _verticies[1]->Parameters();
        Vec3 Vi = param_1.head<3>();
        Vec3 Bai = param_1.segment(3, 3);
        Vec3 Bgi = param_1.tail<3>();

        VecX param_2 = _verticies[2]->Parameters();
        Qd Qj(param_2[6], param_2[3], param_2[4], param_2[5]);
        Vec3 Pj = param_2.head<3>();

        VecX param_3 = _verticies[3]->Parameters();
        Vec3 Vj = param_3.head<3>();
        Vec3 Baj = param_3.segment(3, 3);
        Vec3 Bgj = param_3.tail<3>();

        double sum_dt = _pre_integration->_sum_dt;
        Eigen::Matrix3d dp_dba = _pre_integration->_jacobian.template block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = _pre_integration->_jacobian.template block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = _pre_integration->_jacobian.template block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = _pre_integration->_jacobian.template block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = _pre_integration->_jacobian.template block<3, 3>(O_V, O_BG);

        {
            Eigen::Matrix<double, 15, 6, Eigen::RowMajor> jacobian_pose_i;
            jacobian_pose_i.setZero();

            jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
            jacobian_pose_i.block<3, 3>(O_P, O_R) = Utility::SkewSymmetric(Qi.inverse() * (0.5 * para._G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

            Eigen::Quaterniond corrected_delta_q = _pre_integration->_delta_q * Utility::DeltaQ(dq_dbg * (Bgi - _pre_integration->_linearized_bg));
            jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
            jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::SkewSymmetric(Qi.inverse() * (para._G * sum_dt + Vj - Vi));
            _jacobians[0] = jacobian_pose_i;
        }
        {
            Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speedbias_i;
            jacobian_speed_bias_i.setZero();
            // 之所以要减O_V是因为要放到jacobian_speed_bias_i[15, 9]对应的地方
            jacobian_speed_bias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
            jacobian_speed_bias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
            jacobian_speed_bias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;
            jacobian_speed_bias_i.block<3, 3>(O_R, O_BG - O_V) =
                -Utility::Qleft(Qj.inverse() * Qi * _pre_integration->_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
            jacobian_speed_bias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
            jacobian_speed_bias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
            jacobian_speed_bias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;
            jacobian_speed_bias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();
            jacobian_speed_bias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();
            _jacobians[1] = jacobian_speedbias_i;
        }
        {
            Eigen::Matrix<double, 15, 6, Eigen::RowMajor> jacobian_pose_j;
            jacobian_pose_j.setZero();
            jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();
            Eigen::Quaterniond corrected_delta_q = _pre_integration->_delta_q * Utility::DeltaQ(dq_dbg * (Bgi - _pre_integration->_linearized_bg));
            jacobian_pose_j.block<3, 3>(O_R, O_R) =
                Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
            _jacobians[2] = jacobian_pose_j;
        }
        {
            Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speedbias_j;
            jacobian_speed_bias_j.setZero();
            jacobian_speed_bias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();
            jacobian_speed_bias_j.block<3, 3>(O_BA,O_BA - O_V) = Eigen::Matrix3d::Identity();
            jacobian_speed_bias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();
            _jacobians_[3] = jacobian_speedbias_j;
        }
    }

private:
    enum StateOrder
    {
        O_P = 0,
        O_R = 3,
        O_V = 6,
        O_BA = 9,
        O_BG = 12
    };

    IntegrationBase* _pre_integration;
    static Vec3 _gracity;

    Mat33 _dp_dba = Mat33::Zero();
    Mat33 _dp_dbg = Mat33::Zero();
    Mat33 _dr_dbg = Mat33::Zero();
    Mat33 _dv_dba = Mat33::Zero();
    Mat33 _dv_dbg = Mat33::Zero();
};

#endif //VIO_EXAMPLE_EDGEIMU_H
