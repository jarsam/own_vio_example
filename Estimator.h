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
#include "InitialAlignment.h"
#include "Utility.h"
#include "InitialExRotation.h"
#include "InitialSfM.h"
#include "MotionEstimator.h"

class Estimator
{
public:
    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    Estimator(){
        ClearState();
    }
    void ProcessIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity);
    void ProcessImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header);

private:
    void ClearState(){
        _first_imu = false;
        _solver_flag = INITIAL;
        _initial_timestamp = 0;

        _estimate_extrinsic = svar.GetInt("estimate_extrinsic", 2);
        _Ps.reserve(svar.GetInt("window_size", 20) + 1);
        _Vs.reserve(svar.GetInt("window_size", 20) + 1);
        _Rs.reserve(svar.GetInt("window_size", 20) + 1);
        _Bas.reserve(svar.GetInt("window_size", 20) + 1);
        _Bgs.reserve(svar.GetInt("window_size", 20) + 1);
        _dt_buf.reserve(svar.GetInt("window_size", 20) + 1);
        _linear_acceleration_buf.reserve(svar.GetInt("window_size", 20) + 1);
        _angular_velocity_buf.reserve(svar.GetInt("window_size", 20) + 1);
        _headers.reserve(svar.GetInt("window_size", 20) + 1);
        _pre_integrations.reserve(svar.GetInt("window_size", 20) + 1);
        _ric.reserve(svar.GetInt("number_of_camera", 1));
        _tic.reserve(svar.GetInt("number_of_camera", 1));

        for(int i = 0; i < svar.GetInt("window_size", 20) + 1; ++i){
            _Rs[i].setIdentity();
            _Ps[i].setZero();
            _Vs[i].setZero();
            _Bas[i].setZero();
            _Bgs[i].setZero();

            _frame_count = 0;
        }

        _feature_manager.ClearState();
        _feature_manager = FeatureManager(_Rs);
    }

    bool InitialStructure();
    bool RelativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l);
    bool VisualInitialAlign();

public:
    double _td;

    SolverFlag _solver_flag;
    MarginalizationFlag _marginalization_flag;

    bool _first_imu;
    int _estimate_extrinsic;
    // 滑窗中的帧数
    // 应该是_frame_count = 1的时候才是第一帧.
    double _frame_count;
    double _initial_timestamp;

    std::vector<double> _headers;

    Eigen::Vector3d _g;
    Eigen::Vector3d _acc0, _gyr0;

    std::vector<std::shared_ptr<IntegrationBase>> _pre_integrations;

    std::vector<Eigen::Matrix3d> _ric;
    std::vector<Eigen::Vector3d> _tic;
    std::vector<Eigen::Vector3d> _Ps;
    std::vector<Eigen::Vector3d> _Vs;
    std::vector<Eigen::Matrix3d> _Rs;
    std::vector<Eigen::Vector3d> _Bas;
    std::vector<Eigen::Vector3d> _Bgs;
    std::vector<std::vector<double> > _dt_buf;
    std::vector<std::vector<Eigen::Vector3d> > _linear_acceleration_buf;
    std::vector<std::vector<Eigen::Vector3d> > _angular_velocity_buf;

    FeatureManager _feature_manager;
    InitialExRotation _initial_ex_rotation;
    // 用于在创建ImageFrame对象时,把该指针赋给imageframe.pre_integration.
    std::shared_ptr<IntegrationBase> _tmp_pre_integration;
    MotionEstimator _motion_estimator;

    std::map<double, ImageFrame> _all_image_frame;
};


#endif //VIO_EXAMPLE_ESTIMATOR_H
