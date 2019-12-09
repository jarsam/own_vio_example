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
    if(!_pre_integrations[_frame_count])
        _pre_integrations[_frame_count] = std::shared_ptr<IntegrationBase>(new IntegrationBase(_acc0, _gyr0, _Bas[_frame_count], _Bgs[_frame_count]));
    if(_frame_count != 0){
        _pre_integrations[_frame_count]->PushBack(dt, linear_acceleration, angular_velocity);

    }
}

void Estimator::ProcessImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header)
{
    if (_feature_manager.AddFeatureCheckParallax(_frame_count, image, _td))
        _marginalization_flag = MARGIN_OLD;
    else
        _marginalization_flag = MARGIN_SECOND_NEW;
}