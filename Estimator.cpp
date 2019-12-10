//
// Created by liu on 19-12-5.
//

#include "Estimator.h"

void Estimator::ProcessIMU(double dt, const Eigen::Vector3d &linear_acceleration,
                           const Eigen::Vector3d &angular_velocity)
{
    if (!_first_imu){
        _first_imu = true;
        _acc0 = linear_acceleration;
        _gyr0 = angular_velocity;
    }
    // 当滑窗不满的时候,把当前测量值加入到滑窗指定位置,所以在这个阶段做预积分的时候相当于是在和自己做预积分
    // 在第一帧的时候会有很多Imu信息,这里加入的是最开始的Imu信息.
    if(!_pre_integrations[_frame_count])
        _pre_integrations[_frame_count] = std::shared_ptr<IntegrationBase>(new IntegrationBase(_acc0, _gyr0, _Bas[_frame_count], _Bgs[_frame_count]));
    // 进入预积分阶段
    // 第一帧图像特征点数据没有对应的预积分
    if(_frame_count != 0){
        _pre_integrations[_frame_count]->PushBack(dt, linear_acceleration, angular_velocity);
        _tmp_pre_integration->PushBack(dt, linear_acceleration, angular_velocity);
        _dt_buf[_frame_count].emplace_back(dt);
        _linear_acceleration_buf[_frame_count].emplace_back(linear_acceleration);
        _angular_velocity_buf[_frame_count].emplace_back(angular_velocity);

        // 用Imu数据进行积分,当积完一个measurement中所有Imu数据后,就得到了对应图像帧在世界坐标系的Ps,Vs,Rs
        int j = _frame_count;
        Eigen::Vector3d un_acc_0 = _Rs[j] * (_acc0 - _Bas[j]) - _g;
        Eigen::Vector3d un_gyr = 0.5 * (_gyr0 + angular_velocity) - _Bgs[j];
        _Rs[j] *= Utility::DeltaQ(un_gyr * dt).toRotationMatrix();
        Eigen::Vector3d un_acc_1 = _Rs[j] * (linear_acceleration - _Bas[j]) - _g;
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        _Ps[j] += dt * _Vs[j] + 0.5 * dt * dt * un_acc;
        _Vs[j] += dt * un_acc;
    }
    _acc0 = linear_acceleration;
    _gyr0 = angular_velocity;
}

void Estimator::ProcessImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header)
{
    if (_feature_manager.AddFeatureCheckParallax(_frame_count, image, _td))
        _marginalization_flag = MARGIN_OLD;// KeyFrame
    else
        _marginalization_flag = MARGIN_SECOND_NEW; // Non-KeyFrame

    _headers[_frame_count] = header;

    ImageFrame image_frame(image, header);
    image_frame._pre_integration = _tmp_pre_integration;
    // 每读取一帧图像特征点数据,都会存入_all_image_frame
    _all_image_frame.insert(std::make_pair(header, image_frame));
    _tmp_pre_integration = std::shared_ptr<IntegrationBase>(new IntegrationBase(_acc0, _gyr0, _Bas[_frame_count], _Bgs[_frame_count]));

    // 相机和IMU之间的相对旋转
    if (_estimate_extrinsic == 2){
        if (_frame_count != 0){
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres = _feature_manager.GetCorresponding(_frame_count - 1, _frame_count);
            Eigen::Matrix3d calib_ric;
            if ()
        }
    }

    if (_solver_flag == INITIAL){
        // 只有滑窗中的KeyFrame数量达到指定大小的时候才开始优化.
        if (_frame_count == svar.GetInt("window_size", 10)){

        }
        else
            _frame_count++;
    }
    else{

    }
}