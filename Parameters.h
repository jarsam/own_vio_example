//
// Created by liu on 19-12-4.
//

#ifndef VIO_EXAMPLE_PARAMETERS_H
#define VIO_EXAMPLE_PARAMETERS_H

#include <Eigen/Dense>

#include <vector>

class GlobalParameters
{
private:
    GlobalParameters(){}

public:
    static GlobalParameters& GetInstance(){
        static GlobalParameters instance;
        return instance;
    }

public:
    double _width;
    double _height;
    std::vector<double > _camera_intrinsics;
    std::vector<double > _distortion_coefficients;

    double _acc_noise;
    double _acc_random;
    double _gyr_noise;
    double _gyr_random;
    double _init_depth = 5.0;

    // 相机和Imu之间的位移是读取参数获得的.
    Eigen::Vector3d _Tic;
    Eigen::Matrix3d _Ric;
    // 重力向量
    Eigen::Vector3d _G = {0.0, 0.0, 9.8};

public:
    bool _pub_this_frame;
};

enum ParameterSize
{
    POSE_SIZE = 7,
    SPEED_BIAS = 9,
    FEATURE_SIZE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};

#define para GlobalParameters::GetInstance()

#endif //VIO_EXAMPLE_PARAMETERS_H
