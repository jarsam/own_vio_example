//
// Created by liu on 19-12-5.
//

#ifndef VIO_EXAMPLE_ESTIMATOR_H
#define VIO_EXAMPLE_ESTIMATOR_H

#include <Eigen/Dense>
#include <GSLAM/core/GSLAM.h>

#include <map>
#include <vector>

#include "IntegrationBase.h"

class Estimator
{
public:
    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    Estimator():_first_imu(false){
        _Ps.reserve(svar.GetInt("window_size", 10) + 1);
        _Vs.reserve(svar.GetInt("window_size", 10) + 1);
        _Rs.reserve(svar.GetInt("window_size", 10) + 1);
        _Bas.reserve(svar.GetInt("window_size", 10) + 1);
        _Bgs.reserve(svar.GetInt("window_size", 10) + 1);
        _dt_buf.reserve(svar.GetInt("window_size", 10) + 1);
        _linear_acceleration_buf.reserve(svar.GetInt("window_size", 10) + 1);
        _angular_velocity_buf.reserve(svar.GetInt("window_size", 10) + 1);

        _feature_manager = FeatureManager(_Rs);

        ClearState();
    }
    void ProcessIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity);
    void ProcessImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header);

private:
    void ClearState(){
        for(int i = 0; i < svar.GetInt("window_size", 10) + 1; ++i){
            _Rs[i].setIdentity();
            _Ps[i].setZero();
            _Vs[i].setZero();
            _Bas[i].setZero();
            _Bgs[i].setZero();

            _frame_count = 0;
        }
    }

public:
    double _td;

    SolverFlag _solver_flag;

private:
    bool _first_imu;
    double _frame_count;

    Eigen::Vector3d _acc0, _gyr0;

    std::vector<std::shared_ptr<IntegrationBase>> _pre_integrations;

    std::vector<Eigen::Vector3d> _Ps;
    std::vector<Eigen::Vector3d> _Vs;
    std::vector<Eigen::Matrix3d> _Rs;
    std::vector<Eigen::Vector3d> _Bas;
    std::vector<Eigen::Vector3d> _Bgs;
    std::vector<std::vector<double> > _dt_buf;
    std::vector<std::vector<Eigen::Vector3d> > _linear_acceleration_buf;
    std::vector<std::vector<Eigen::Vector3d> > _angular_velocity_buf;

    FeatureManager _feature_manager;
};


#endif //VIO_EXAMPLE_ESTIMATOR_H
