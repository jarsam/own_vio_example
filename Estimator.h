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
#include "MarginalizationFactor.h"
#include "ImuFactor.h"
#include "ProjectionTdFactor.h"
#include "ProjectionFactor.h"

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
        _failure_occur = false;
        _relocalization_info = false;
        _first_imu = false;
        _solver_flag = INITIAL;
        _initial_timestamp = 0;
        _frame_count = 0;
        _sum_of_front = 0;
        _sum_of_back = 0;

        _last_marginalization_info = nullptr;
        _last_marginalization_parameter_blocks.clear();

        _estimate_extrinsic = svar.GetInt("estimate_extrinsic", 2);
        _Ps.resize(svar.GetInt("window_size", 20) + 1, Eigen::Vector3d::Zero());
        _Vs.resize(svar.GetInt("window_size", 20) + 1, Eigen::Vector3d::Zero());
        _Rs.resize(svar.GetInt("window_size", 20) + 1, Eigen::Matrix3d::Zero());
        _Bas.resize(svar.GetInt("window_size", 20) + 1, Eigen::Vector3d::Zero());
        _Bgs.resize(svar.GetInt("window_size", 20) + 1, Eigen::Vector3d::Zero());
        _dt_buf.resize(svar.GetInt("window_size", 20) + 1, std::vector<double>());
        _linear_acceleration_buf.resize(svar.GetInt("window_size", 20) + 1, std::vector<Eigen::Vector3d>());
        _angular_velocity_buf.resize(svar.GetInt("window_size", 20) + 1, std::vector<Eigen::Vector3d>());
        _headers.resize(svar.GetInt("window_size", 20) + 1, 0);
        _pre_integrations.resize(svar.GetInt("window_size", 20) + 1, nullptr);
        _ric.resize(svar.GetInt("camera_number", 1), Eigen::Matrix3d::Zero());
        _tic.resize(svar.GetInt("camera_number", 1), Eigen::Vector3d::Zero());

        _para_pose.resize(svar.GetInt("window_size", 20) + 1, std::vector<double>(POSE_SIZE, 0));
        _para_speed_bias.resize(svar.GetInt("window_size", 20) + 1, std::vector<double>(SPEED_BIAS, 0));
        _para_feature.resize(svar.GetInt("feature_number", 1000), std::vector<double>(FEATURE_SIZE, 0));
        _para_ex_pose.resize(svar.GetInt("camera_number", 1), std::vector<double>(POSE_SIZE, 0));
        _para_retrive_pose.resize(POSE_SIZE, 0.0);
        _relo_pose.resize(POSE_SIZE, 0.0);
        _para_td.resize(1, std::vector<double>(1, 0));
        _para_tr.resize(1, std::vector<double>(1, 0));

        _tmp_pre_integration = nullptr;
        for(int i = 0; i < svar.GetInt("window_size", 20) + 1; ++i){
            _Rs[i].setIdentity();
            _Ps[i].setZero();
            _Vs[i].setZero();
            _Bas[i].setZero();
            _Bgs[i].setZero();
            _pre_integrations[i] = nullptr;
        }

        for(auto &it: _all_image_frame){
            if (it.second._pre_integration != nullptr){
                it.second._pre_integration = nullptr;
            }
        }

        _feature_manager.ClearState();
        _feature_manager = FeatureManager(_Rs);

        _all_image_frame.clear();
    }

    bool InitialStructure();
    bool RelativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l);
    bool VisualInitialAlign();
    void SlideWindow();
    void SlideWindowOld();
    void SlideWindowNew();
    void SolveOdometry();
    void BackendOptimization();
    void Vector2Double();
    void Double2Vector();
    bool FailureDetection();
    void SetParameter();

public:
    SolverFlag _solver_flag;
    MarginalizationFlag _marginalization_flag;

    bool _failure_occur;
    bool _relocalization_info;
    bool _first_imu;
    int _estimate_extrinsic;
    int _sum_of_back, _sum_of_front;// Margin_Old和Margin_New的次数
    // 滑窗中的帧数
    // 应该是_frame_count = 1的时候才是第一帧.
    int _frame_count;
    int _relo_frame_local_index;
    double _initial_timestamp, _td;

    std::vector<double> _headers;
    std::vector<double> _para_retrive_pose;
    std::vector<double> _relo_pose;
    std::vector<std::vector<double> > _para_pose;
    std::vector<std::vector<double> > _para_speed_bias;
    std::vector<std::vector<double> > _para_feature;
    std::vector<std::vector<double> > _para_ex_pose;
    std::vector<std::vector<double> > _para_td;
    std::vector<std::vector<double> > _para_tr;

    Eigen::Vector3d _g;
    Eigen::Vector3d _acc0, _gyr0;// 最近一次接收到的Imu数据
    // _backR0, _backP0是MARGIN的帧的位姿.
    Eigen::Matrix3d _backR0, _lastR, _lastR0;
    Eigen::Vector3d _backP0, _lastP, _lastP0;

    std::vector<std::shared_ptr<IntegrationBase>> _pre_integrations;

    std::vector<Eigen::Vector3d> _key_poses;
    std::vector<Eigen::Vector3d> _match_points;
    std::vector<Eigen::Matrix3d> _ric;// 从相机到Imu的旋转
    std::vector<Eigen::Vector3d> _tic;// 从相机到Imu的平移
    std::vector<Eigen::Vector3d> _Ps;// 滑动窗口中各帧在世界坐标系下的位置
    std::vector<Eigen::Vector3d> _Vs;// 滑动窗口中各帧在世界坐标系下的速度
    std::vector<Eigen::Matrix3d> _Rs;// 滑动窗口中各帧在世界坐标系下的旋转
    std::vector<Eigen::Vector3d> _Bas;// 滑动窗口中各帧对应的加速度偏置
    std::vector<Eigen::Vector3d> _Bgs;// 滑动窗口中各帧对应的陀螺仪偏置
    std::vector<std::vector<double> > _dt_buf;
    std::vector<std::vector<Eigen::Vector3d> > _linear_acceleration_buf;// 滑动窗口中的传入的Imu加速度
    std::vector<std::vector<Eigen::Vector3d> > _angular_velocity_buf;// 滑动窗口中的传入的Imu角速度

    MotionEstimator _motion_estimator;
    FeatureManager _feature_manager;
    InitialExRotation _initial_ex_rotation;
    // 用于在创建ImageFrame对象时,把该指针赋给imageframe.pre_integration.
    std::shared_ptr<IntegrationBase> _tmp_pre_integration;
    // 上一次边缘化的信息.
    std::shared_ptr<MarginalizationInfo> _last_marginalization_info;
    std::vector<double *> _last_marginalization_parameter_blocks;

    std::map<double, ImageFrame> _all_image_frame;
};


#endif //VIO_EXAMPLE_ESTIMATOR_H
