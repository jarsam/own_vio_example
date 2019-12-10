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
    // FIXME: 第一帧的Imu信息不会进入预积分阶段
    if(_frame_count != 0){
        _pre_integrations[_frame_count]->PushBack(dt, linear_acceleration, angular_velocity);
        _tmp_pre_integration->PushBack(dt, linear_acceleration, angular_velocity);
        _dt_buf[_frame_count].emplace_back(dt);
        _linear_acceleration_buf[_frame_count].emplace_back(linear_acceleration);
        _angular_velocity_buf[_frame_count].emplace_back(angular_velocity);

        int j = _frame_count;

        Eigen::Vector3d un_acc_0 = _Rs[j] * (_acc0 - _Bas[j]) - _g;
        Eigen::Vector3d un_gyr = 0.5 * (_gyr0 + angular_velocity) - _Bgs[j];
    }
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
    _all_image_frame.insert(std::make_pair(header, image_frame));
    _tmp_pre_integration = std::shared_ptr<IntegrationBase>(new IntegrationBase(_acc0, _gyr0, _Bas[_frame_count], _Bgs[_frame_count]));

    // 相机和IMU之间的相对旋转
    if (_estimate_extrinsic == 2){
        if (_frame_count != 0){

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