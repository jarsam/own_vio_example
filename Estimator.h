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
        _Ps.resize(svar.GetInt("window_size", 20) + 1);
        _Vs.resize(svar.GetInt("window_size", 20) + 1);
        _Rs.resize(svar.GetInt("window_size", 20) + 1);
        _Bas.resize(svar.GetInt("window_size", 20) + 1);
        _Bgs.resize(svar.GetInt("window_size", 20) + 1);
        _dt_buf.resize(svar.GetInt("window_size", 20) + 1);
        _linear_acceleration_buf.resize(svar.GetInt("window_size", 20) + 1);
        _angular_velocity_buf.resize(svar.GetInt("window_size", 20) + 1);
        _headers.resize(svar.GetInt("window_size", 20) + 1);
        _pre_integrations.resize(svar.GetInt("window_size", 20) + 1, nullptr);
        _ric.resize(svar.GetInt("number_of_camera", 1));
        _tic.resize(svar.GetInt("number_of_camera", 1));

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
    void SlideWindow();
    void SlideWindowOld();

public:
    SolverFlag _solver_flag;
    MarginalizationFlag _marginalization_flag;

    bool _first_imu;
    int _estimate_extrinsic;
    int _sum_of_back, _sum_of_front;
    // 滑窗中的帧数
    // 应该是_frame_count = 1的时候才是第一帧.
    int _frame_count;
    double _initial_timestamp;
    double _td;

    std::vector<double> _headers;

    Eigen::Vector3d _g;
    Eigen::Vector3d _acc0, _gyr0;// 最近一次接收到的Imu数据
    // _backR0, _backP0是MARGIN的帧的位姿.
    Eigen::Matrix3d _backR0, _lastR, _lastR0;
    Eigen::Vector3d _backP0, _lastP, _lastP0;

    std::vector<std::shared_ptr<IntegrationBase>> _pre_integrations;

    std::vector<Eigen::Matrix3d> _ric;// 从相机到Imu的旋转
    std::vector<Eigen::Vector3d> _tic;// 从相机到Imu的平移
    std::vector<Eigen::Vector3d> _Ps;// 滑动窗口中各帧在世界坐标系下的位置
    std::vector<Eigen::Vector3d> _Vs;// 滑动窗口中各帧在世界坐标系下的速度
    std::vector<Eigen::Matrix3d> _Rs;// 滑动窗口中各帧在世界坐标系下的旋转
    std::vector<Eigen::Vector3d> _Bas;// 滑动窗口中各帧对应的加速度偏置
    std::vector<Eigen::Vector3d> _Bgs;// 滑动窗口中各帧对应的陀螺仪偏置
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
