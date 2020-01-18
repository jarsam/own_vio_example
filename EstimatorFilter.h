//
// Created by liu on 20-1-15.
//

#ifndef VIO_EXAMPLE_ESTIMATORFILTER_H
#define VIO_EXAMPLE_ESTIMATORFILTER_H

#include <Eigen/Dense>
#include <GSLAM/core/GSLAM.h>
#include <boost/shared_ptr.hpp>
#include <boost/math/distributions/chi_squared.hpp>

#include <map>
#include <vector>

#include "Estimator.h"
#include "Parameters.h"

typedef long long int StateIDType;

struct ImuState
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    StateIDType _id;
    static StateIDType _next_id;
    double _time;

    // 世界坐标系到Imu坐标系
    Eigen::Vector4d _orientation;
    Eigen::Vector3d _position;
    Eigen::Vector3d _velocity;
    Eigen::Vector3d _gyro_bias;
    Eigen::Vector3d _acc_bias;

    Eigen::Matrix3d _R_imu_cam0;
    Eigen::Vector3d _t_cam0_imu;

    Eigen::Vector4d _orientation_null;
    Eigen::Vector3d _position_null;
    Eigen::Vector3d _velocity_null;

    static double _gyro_noise;
    static double _acc_noise;
    static double _gyro_bias_noise;
    static double _acc_bias_noise;

    static Eigen::Vector3d _gravity;
    ImuState():_id(0), _time(0), _orientation(Eigen::Vector4d(0, 0, 0, 1)),
               _position(Eigen::Vector3d::Zero()),
               _velocity(Eigen::Vector3d::Zero()),
               _gyro_bias(Eigen::Vector3d::Zero()),
               _acc_bias(Eigen::Vector3d::Zero()),
               _orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
               _position_null(Eigen::Vector3d::Zero()),
               _velocity_null(Eigen::Vector3d::Zero()){}
    ImuState(const StateIDType& new_id): _id(new_id), _time(0), _orientation(Eigen::Vector4d(0, 0, 0, 1)),
                                         _position(Eigen::Vector3d::Zero()),
                                         _velocity(Eigen::Vector3d::Zero()),
                                         _gyro_bias(Eigen::Vector3d::Zero()),
                                         _acc_bias(Eigen::Vector3d::Zero()),
                                         _orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
                                         _position_null(Eigen::Vector3d::Zero()),
                                         _velocity_null(Eigen::Vector3d::Zero()){}
};

struct CamState
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    StateIDType _id;
    double _time;

    // 世界坐标系到相机坐标系
    Eigen::Vector4d _orientation;
    Eigen::Vector3d _position;
    Eigen::Vector3d _orientation_null;
    Eigen::Vector3d _position_null;

    CamState(): _id(0), _time(0), _orientation(Eigen::Vector4d(0, 0, 0, 1)), _position(Eigen::Vector3d::Zero()),
                _orientation_null(Eigen::Vector4d(0, 0, 0, 1)), _position_null(Eigen::Vector3d(0, 0, 0)) {}
    CamState(const StateIDType& new_id ): _id(new_id), _time(0),
                                          _orientation(Eigen::Vector4d(0, 0, 0, 1)),
                                          _position(Eigen::Vector3d::Zero()),
                                          _orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
                                          _position_null(Eigen::Vector3d::Zero()) {}
};

struct StateServer
{
    StateServer(){}
    ImuState _imu_state;
    CamState _cam_state;

    Eigen::MatrixXd _state_cov;
    Eigen::Matrix<double, 12, 12> _continuous_noise_cov;
};

class EstimatorFilter: public Estimator
{
public:
    EstimatorFilter(){
        Initialize();
    }
    ~EstimatorFilter(){}

    void LoadParameters();
    void Initialize();

    virtual void ProcessIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity);
    virtual void ProcessImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header);
    virtual void ClearState();

private:

public:
    StateServer _state_server;

    static std::map<int, double> _chi_squared_test_table;
};


#endif //VIO_EXAMPLE_ESTIMATORFILTER_H
