//
// Created by liu on 19-12-18.
//

#ifndef VIO_EXAMPLE_IMUFACTOR_H
#define VIO_EXAMPLE_IMUFACTOR_H

#include <iostream>

#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "IntegrationBase.h"
#include "Utility.h"

class ImuFactor: public ceres::SizedCostFunction<15, 7, 9, 7, 9>
{
public:
    ImuFactor() = delete;
    ImuFactor(std::shared_ptr<IntegrationBase> pre_integration): _pre_integration(pre_integration){}

    virtual bool Evaluate(double const *const *parameters, double* residuals, double **jacobians)const
    {
        Eigen::Vector3d    Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d    Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d   Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d   Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

        Eigen::Vector3d    Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        Eigen::Vector3d    Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Vector3d   Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
        Eigen::Vector3d   Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

        Eigen::Map<Eigen::Matrix<double, 15, 1> > residual(residuals);
        // 在优化迭代的过程中, 预积分值是不变的, 输入的状态值会被不断的更新, 然后不断的调用Evaluate()计算更新后的Imu残差.
        residual = _pre_integration->Evaluate(Pi, Qi, Vi, Bai, Bgi,
                                              Pj, Qj, Vj, Baj, Bgj);
        Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15> >(
            _pre_integration->_covariance.inverse()).matrixL().transpose();
        // 为了保证Imu和视觉残差项在尺度上保持一致, 一般会采用与量纲无关的马氏距离.
        residual = sqrt_info * residual;

        if (jacobians){
            double sum_dt = _pre_integration->_sum_dt;
            Eigen::Matrix3d dp_dba = _pre_integration->_jacobian.template block<3, 3>(O_P, O_BA);
            Eigen::Matrix3d dp_dbg = _pre_integration->_jacobian.template block<3, 3>(O_P, O_BG);
            Eigen::Matrix3d dq_dbg = _pre_integration->_jacobian.template block<3, 3>(O_R, O_BG);
            Eigen::Matrix3d dv_dba = _pre_integration->_jacobian.template block<3, 3>(O_V, O_BA);
            Eigen::Matrix3d dv_dbg = _pre_integration->_jacobian.template block<3, 3>(O_V, O_BG);

            if (jacobians[0]){
                // 要注意这里的旋转矩阵正逆关系
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor> > jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
                jacobian_pose_i.block<3, 3>(O_P, O_R) =
                    Utility::SkewSymmetric(Qi.inverse() * (0.5 * para._G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
                Eigen::Quaterniond corrected_delta_q =
                    _pre_integration->_delta_q * Utility::DeltaQ(dq_dbg * (Bgi - _pre_integration->_linearized_bg));
                jacobian_pose_i.block<3, 3>(O_V, O_R) =
                    Utility::SkewSymmetric(Qi.inverse() * (para._G * sum_dt + Vj - Vi));
                // 注意到这里同样乘了sqrt_info
                jacobian_pose_i = sqrt_info * jacobian_pose_i;
            }
        }
    }

    std::shared_ptr<IntegrationBase> _pre_integration;
};

#endif //VIO_EXAMPLE_IMUFACTOR_H
