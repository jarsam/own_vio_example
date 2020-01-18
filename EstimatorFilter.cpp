//
// Created by liu on 20-1-15.
//

#include "EstimatorFilter.h"

void EstimatorFilter::LoadParameters()
{
    _state_server._imu_state._gyro_noise = para._gyr_noise * para._gyr_noise;
    _state_server._imu_state._gyro_bias_noise = para._gyr_random * para._gyr_random;
    _state_server._imu_state._acc_noise = para._acc_noise * para._acc_noise;
    _state_server._imu_state._acc_bias_noise = para._acc_random * para._acc_random;

    _state_server._continuous_noise_cov = Eigen::Matrix<double, 12, 12>::Zero();
    _state_server._continuous_noise_cov.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * _state_server._imu_state._gyro_noise;
    _state_server._continuous_noise_cov.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * _state_server._imu_state._gyro_bias_noise;
    _state_server._continuous_noise_cov.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * _state_server._imu_state._acc_noise;
    _state_server._continuous_noise_cov.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * _state_server._imu_state._acc_bias_noise;

    // 设置Imu的初始协方差
    // 方向和位置的协方差可以设置为0
    // 速度, 偏置以及外参数应有不确定性(协方差应该给初始值)
    double gyro_bias_cov = 1e-4, acc_bias_cov = 1e-2, velocity_cov = 0.25;
    double extrinsic_rotation_cov = 3.0462e-4, extrinsic_translation_cov = 1e-4;
    // 连续时间下的状态协方差矩阵初始值P0
    // 协方差的维度为21*21，其中分别对应对应状态[q b_g v b_a p q_e p_e]
    _state_server._state_cov = Eigen::MatrixXd::Zero(21, 21);
    for(int i = 3; i < 6; ++i)
        _state_server._state_cov(i, i) = gyro_bias_cov;
    for(int i = 6; i < 9; ++i)
        _state_server._state_cov(i, i) = velocity_cov;
    for(int i = 9; i < 12; ++i)
        _state_server._state_cov(i, i) = acc_bias_cov;
    for(int i = 15; i < 18; ++i)
        _state_server._state_cov(i, i) = extrinsic_rotation_cov;
    for(int i = 18; i < 21; ++i)
        _state_server._state_cov(i, i) = extrinsic_translation_cov;
}

void EstimatorFilter::Initialize()
{
    LoadParameters();

    for(int i = 1; i < 100; ++i){
        boost::math::chi_squared chi_squared_dist(i);
        _chi_squared_test_table[i] = boost::math::quantile(chi_squared_dist, 0.05);
    }
}

void EstimatorFilter::ProcessImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                                   double header)
{
    
}

void EstimatorFilter::ProcessIMU(double dt, const Eigen::Vector3d &linear_acceleration,
                                 const Eigen::Vector3d &angular_velocity)
{
    ImuState& imu_state = _state_server._imu_state;
    Eigen::Vector3d gyro = angular_velocity - imu_state._gyro_bias;
    Eigen::Vector3d acc = linear_acceleration - imu_state._acc_bias;
    double dtime = dt;

    Eigen::Matrix<double, 21, 21> F = Eigen::Matrix<double, 21, 21>::Zero();
    Eigen::Matrix<double, 21, 21> G = Eigen::Matrix<double, 21, 21>::Zero();
}

void EstimatorFilter::ClearState()
{

}

