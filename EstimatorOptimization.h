//
// Created by liu on 20-1-15.
//

#ifndef VIO_EXAMPLE_ESTIMATOROPTIMIZATION_H
#define VIO_EXAMPLE_ESTIMATOROPTIMIZATION_H

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
#include "Problem.h"
#include "VertexPose.h"
#include "VertexSpeedBias.h"
#include "VertexInverseDepth.h"
#include "EdgeImu.h"
#include "EdgePrior.h"
#include "EdgeReprojection.h"
#include "Estimator.h"

class EstimatorOptimization: public Estimator
{
public:
    EstimatorOptimization(){
        ClearState();
    }
    ~EstimatorOptimization(){
        for(int i = 0; i < svar.GetInt("window_size") + 1; ++i)
            delete[] _para_pose[i];
        delete _para_pose;

        for(int i = 0; i < svar.GetInt("window_size") + 1; ++i)
            delete[] _para_speed_bias[i];
        delete _para_speed_bias;

        for(int i = 0; i < svar.GetInt("feature_number"); ++i)
            delete[] _para_feature[i];
        delete _para_feature;

        for(int i = 0; i < svar.GetInt("camera_number"); ++i)
            delete[] _para_ex_pose[i];
        delete _para_ex_pose;

        delete[] _para_retrive_pose;

        delete[] _para_td[0];
        delete _para_td;

        delete[] _para_tr[0];
        delete _para_tr;
    }
    virtual void ProcessIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity);
    virtual void ProcessImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header);
    virtual void ClearState(){
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
        _ric.resize(svar.GetInt("camera_number", 1), Eigen::Matrix3d::Zero());
        _tic.resize(svar.GetInt("camera_number", 1), Eigen::Vector3d::Zero());

        _para_pose = new double*[svar.GetInt("window_size") + 1];
        for(int i = 0; i < svar.GetInt("window_size") + 1; ++i)
            _para_pose[i] = new double[POSE_SIZE];

        _para_speed_bias = new double*[svar.GetInt("window_size") + 1];
        for(int i = 0; i < svar.GetInt("window_size") + 1; ++i)
            _para_speed_bias[i] = new double[SPEED_BIAS];

        _para_feature = new double*[svar.GetInt("feature_number")];
        for(int i = 0; i < svar.GetInt("feature_number"); ++i)
            _para_feature[i] = new double[FEATURE_SIZE];

        _para_ex_pose = new double*[svar.GetInt("camera_number")];
        for(int i = 0; i < svar.GetInt("camera_number"); ++i)
            _para_ex_pose[i] = new double[POSE_SIZE];

        _para_retrive_pose = new double[POSE_SIZE];

        _para_td = new double*[1];
        _para_td[0] = new double[1];

        _para_tr = new double*[1];
        _para_tr[0] = new double[1];

        _pre_integrations = new IntegrationBase*[svar.GetInt("window_size") + 1];
        for(int i = 0; i < svar.GetInt("window_size") + 1; ++i)
            _pre_integrations[i] = nullptr;

        _tmp_pre_integration = nullptr;
        for(int i = 0; i < svar.GetInt("window_size", 20) + 1; ++i){
            _Rs[i].setIdentity();
            _Ps[i].setZero();
            _Vs[i].setZero();
            _Bas[i].setZero();
            _Bgs[i].setZero();
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
    void BackendOptimizationCeres();
    void BackendOptimizationEigen();
    void Vector2Double();
    void Double2Vector();
    bool FailureDetection();
    void SetParameter();
    void ProblemSolve();
    void MarginOldFrame();
    void MarginNewFrame();

public:
    bool _failure_occur;
    bool _relocalization_info;
    bool _first_imu;
    int _estimate_extrinsic;
    int _sum_of_back, _sum_of_front;// Margin_Old和Margin_New的次数
    // 滑窗中的帧数
    // 应该是_frame_count = 1的时候才是第一帧.
    int _frame_count;
    int _relo_frame_local_index;

    std::vector<double> _relo_pose;

    double** _para_pose;
    double** _para_speed_bias;
    double** _para_feature;
    double** _para_ex_pose;
    double* _para_retrive_pose;
    double** _para_td;
    double** _para_tr;

    IntegrationBase **_pre_integrations;

    Eigen::Vector3d _g;
    Eigen::Vector3d _acc0, _gyr0;// 最近一次接收到的Imu数据
    // _backR0, _backP0是MARGIN的帧的位姿.
    Eigen::Matrix3d _backR0, _lastR, _lastR0;
    Eigen::Vector3d _backP0, _lastP, _lastP0;

    std::vector<Eigen::Vector3d> _key_poses;
    std::vector<Eigen::Vector3d> _match_points;
    std::vector<Eigen::Matrix3d> _ric;// 从相机到Imu的旋转
    std::vector<Eigen::Vector3d> _tic;// 从相机到Imu的平移
    std::vector<std::vector<double> > _dt_buf;
    std::vector<std::vector<Eigen::Vector3d> > _linear_acceleration_buf;// 滑动窗口中的传入的Imu加速度
    std::vector<std::vector<Eigen::Vector3d> > _angular_velocity_buf;// 滑动窗口中的传入的Imu角速度

    MotionEstimator _motion_estimator;
    FeatureManager _feature_manager;
    InitialExRotation _initial_ex_rotation;
    // 用于在创建ImageFrame对象时,把该指针赋给imageframe.pre_integration.
    IntegrationBase* _tmp_pre_integration;
    // 上一次边缘化的信息.
    MarginalizationInfo* _last_marginalization_info;
    // 之前边缘化的参数块, 已经把要marg 的参数块去除了.
    std::vector<double *> _last_marginalization_parameter_blocks;

    std::map<double, ImageFrame> _all_image_frame;

    //////////////// OUR SOLVER ///////////////////
    MatXX _H_prior;
    VecX _b_prior;
    VecX _err_prior;
    MatXX _J_prior_inv;

    Eigen::Matrix2d _project_sqrt_info;
    //////////////// OUR SOLVER //////////////////
};


#endif //VIO_EXAMPLE_ESTIMATOROPTIMIZATION_H
