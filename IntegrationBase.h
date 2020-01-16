//
// Created by liu on 19-12-6.
//

#ifndef VIO_EXAMPLE_INTEGRATIONBASE_H
#define VIO_EXAMPLE_INTEGRATIONBASE_H

#include <ceres/ceres.h>

#include "Parameters.h"
#include "FeatureManager.h"
#include "Utility.h"

class IntegrationBase
{
public:
    IntegrationBase() = delete;
    IntegrationBase(const Eigen::Vector3d &acc_0, const Eigen::Vector3d &gyr_0,
                     const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg)
                     : _acc0(acc_0), _gyr0(gyr_0), _linearized_acc(acc_0), _linearized_gyr(gyr_0),
                       _linearized_ba(linearized_ba), _linearized_bg(linearized_bg),
                       _jacobian(Eigen::Matrix<double, 15, 15>::Identity()), _covariance(Eigen::Matrix<double, 15, 15>::Zero()),
                       _sum_dt(0.0), _delta_p(Eigen::Vector3d::Zero()), _delta_q(Eigen::Quaterniond::Identity()),
                       _delta_v(Eigen::Vector3d::Zero())
    {
        // 如果噪声之间没有什么关系的话,初始化如下.
        _noise = Eigen::Matrix<double, 18, 18>::Zero();
        _noise.block<3, 3>(0, 0) = (para._acc_noise * para._acc_noise) * Eigen::Matrix3d::Identity();
        _noise.block<3, 3>(3, 3) = (para._gyr_noise * para._gyr_noise) * Eigen::Matrix3d::Identity();
        _noise.block<3, 3>(6, 6) = (para._acc_noise * para._acc_noise) * Eigen::Matrix3d::Identity();
        _noise.block<3, 3>(9, 9) = (para._gyr_noise * para._gyr_noise) * Eigen::Matrix3d::Identity();
        _noise.block<3, 3>(12, 12) = (para._acc_random * para._acc_random) * Eigen::Matrix3d::Identity();
        _noise.block<3, 3>(15, 15) = (para._gyr_random * para._gyr_random) * Eigen::Matrix3d::Identity();
    }

    void PushBack(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr){
        _dt_buf.emplace_back(dt);
        _acc_buf.emplace_back(acc);
        _gyr_buf.emplace_back(gyr);
        Propagate(dt, acc, gyr);
    }

    void Propagate(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr){
        _dt = dt;
        _acc1 = acc;
        _gyr1 = gyr;
        Eigen::Vector3d result_delta_p, result_delta_v, result_linearized_ba, result_linearized_bg;
        Eigen::Quaterniond result_delta_q;
        MidPointIntegration(dt, _acc0, _gyr0, _acc1, _gyr1, _delta_p, _delta_q, _delta_v,
                            _linearized_ba, _linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        _delta_p = result_delta_p;
        _delta_q = result_delta_q;
        _delta_v = result_delta_v;
        _linearized_ba = result_linearized_ba;
        _linearized_bg = result_linearized_bg;
        _delta_q.normalize();
        _sum_dt += _dt;
        _acc0 = _acc1;
        _gyr0 = _gyr1;
    }

    void Repropagate(const Eigen::Vector3d& linearized_ba, const Eigen::Vector3d& linearized_bg){
        _sum_dt = 0.0;
        _acc0 = _linearized_acc;
        _gyr0 = _linearized_gyr;
        _delta_p.setZero();
        _delta_q.setIdentity();
        _delta_v.setZero();
        _linearized_ba = linearized_ba;
        _linearized_bg = linearized_bg;
        _jacobian.setIdentity();
        _covariance.setZero();
        for(int i = 0;i < _dt_buf.size(); ++i)
            Propagate(_dt_buf[i], _acc_buf[i], _gyr_buf[i]);
    }

    void MidPointIntegration(double dt, const Eigen::Vector3d &acc0, const Eigen::Vector3d &gyr0,
                             const Eigen::Vector3d &acc1, const Eigen::Vector3d &gyr1,
                             const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                             const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                             Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                             Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian){
        Eigen::Vector3d un_gyr = 0.5 * (gyr0 + gyr1) - linearized_bg;
        result_delta_q = delta_q * Eigen::Quaterniond(1, un_gyr(0) * dt / 2, un_gyr(1) * dt / 2, un_gyr(2) * dt / 2);

        // 转化为世界坐标系下的数据.
        Eigen::Vector3d un_acc0 = delta_q * (acc0 - linearized_ba);
        Eigen::Vector3d un_acc1 = delta_q * (acc1 - linearized_ba);
        Eigen::Vector3d un_acc = 0.5 * (un_acc0 + un_acc1);

        result_delta_p = delta_p + delta_v * dt + 0.5 * un_acc * dt * dt;
        result_delta_v = delta_v + un_acc * dt;

        // 预积分的过程中bias没有发生变化.
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;

        //! 计算雅克比矩阵
        // 离散状态下在计算协方差矩阵的时候为：P' = FPF' + GQG'
        // F矩阵的行顺序为：P, Q, V, Ba, Bg
        if (update_jacobian){
            Eigen::Vector3d w_x = 0.5 * (gyr0 + gyr1) - linearized_bg;
            Eigen::Vector3d a_0_x = acc0 - linearized_ba;
            Eigen::Vector3d a_1_x = acc1 - linearized_ba;

            Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;

            R_w_x<<     0, -w_x(2),  w_x(1),
                w_x(2),      0, -w_x(0),
                -w_x(1), w_x(0),       0;

            R_a_0_x<<        0, -a_0_x(2),  a_0_x(1),
                a_0_x(2),         0, -a_0_x(0),
                -a_0_x(1),  a_0_x(0),         0;

            R_a_1_x<<        0, -a_1_x(2),  a_1_x(1),
                a_1_x(2),         0, -a_1_x(0),
                -a_1_x(1),  a_1_x(0),         0;

            Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0)   = Eigen::Matrix3d::Identity();
            F.block<3, 3>(0, 3)   = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
                                    -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            F.block<3, 3>(0, 6)   = Eigen::MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(0, 9)   = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 12)  = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3)   = Eigen::Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 12)  = -1.0 * Eigen::MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(6, 3)   = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt +
                                    -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6)   = Eigen::Matrix3d::Identity();
            F.block<3, 3>(6, 9)   = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 12)  = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
            F.block<3, 3>(9, 9)   = Eigen::Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();

            Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15,18);
            V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6) =  0.25 *  result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) =  0.5 * Eigen::MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(3, 9) =  0.5 * Eigen::MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6) =  0.5 *  result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12)  = Eigen::MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(12, 15) = Eigen::MatrixXd::Identity(3,3) * _dt;

            _jacobian = F * _jacobian;
            _covariance = F * _covariance * F.transpose() + V * _noise * V.transpose();
        }
    };

    // Pi, Qi, Vi, Bai, Bgi: 前一次预积分结果
    // Pj, Qj, Vj, Baj, Bgj: 后一次预积分结果
    // 这个函数是求Residual的
    Eigen::Matrix<double, 15, 1> Evaluate(
        const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
        const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj
        )
    {
        Eigen::Matrix<double, 15, 1> residuals;
        Eigen::Matrix3d dp_dba = _jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = _jacobian.block<3, 3>(O_P, O_BG);
        Eigen::Matrix3d dq_dbg = _jacobian.block<3, 3>(O_R, O_BG);
        Eigen::Matrix3d dv_dba = _jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = _jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - _linearized_ba;
        Eigen::Vector3d dbg = Bgi - _linearized_bg;

        // IMU预积分的结果,消除掉acc bias和gyro bias的影响, 对应IMU model中的\hat{\alpha},\hat{\beta},\hat{\gamma}
        Eigen::Quaterniond corrected_delta_q = _delta_q * Utility::DeltaQ(dq_dbg * dbg);
        Eigen::Vector3d    corrected_delta_v = _delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d    corrected_delta_p = _delta_p + dp_dba * dba + dp_dbg * dbg;

        // FIXME: 这里用了原本的G, 没用优化后的重力向量?
        // IMU项residual计算,输入参数是状态的估计值, 上面correct_delta_*是预积分值, 二者求'diff'得到residual
        residuals.block<3, 1>(O_P, 0)  = Qi.inverse() * (0.5 * para._G * _sum_dt * _sum_dt + Pj - Pi - Vi * _sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0)  = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0)  = Qi.inverse() * (para._G * _sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;

        return residuals;
    }

public:
    // 当前传入的时间
    double _dt;
    double _sum_dt;

    // 上一帧Imu的信息.
    Eigen::Vector3d _acc0, _gyr0;
    // 当前传入的Imu信息.
    Eigen::Vector3d _acc1, _gyr1;
    // 存的第一帧的Imu信息, 在repropagete的时候使用
    Eigen::Vector3d _linearized_acc, _linearized_gyr;
    // 在最开始的时候,这些参数都是0, 是减去这些噪声.
    // 然后每传入一帧都会预积分算出新的bias.
    // 在初始化的时候, 陀螺仪的bias会进行标定, 会传入第一帧的_linearized_bg, 重新repropagate
    Eigen::Vector3d _linearized_ba, _linearized_bg;
    // 从最开始的变化量
    Eigen::Vector3d _delta_p, _delta_v;
    Eigen::Quaterniond _delta_q;

    Eigen::Matrix<double, 15, 15> _jacobian, _covariance;
    Eigen::Matrix<double, 15, 15> _step_jacobian;
    Eigen::Matrix<double, 15, 18> _step_v;
    Eigen::Matrix<double, 18, 18> _noise;

    std::vector<double> _dt_buf;
    std::vector<Eigen::Vector3d> _acc_buf;
    std::vector<Eigen::Vector3d> _gyr_buf;
};

#endif //VIO_EXAMPLE_INTEGRATIONBASE_H
