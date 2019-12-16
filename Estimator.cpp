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
    // 每读取一帧新的图像都会new一个_tmp_pre_integration.
    _tmp_pre_integration = std::shared_ptr<IntegrationBase>(new IntegrationBase(_acc0, _gyr0, _Bas[_frame_count], _Bgs[_frame_count]));

    // 相机和IMU之间的相对旋转
    if (_estimate_extrinsic == 2){
        if (_frame_count != 0){
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres = _feature_manager.GetCorresponding(_frame_count - 1, _frame_count);
            Eigen::Matrix3d calib_ric;
            if (_initial_ex_rotation.CalibrationExRotation(corres, _pre_integrations[_frame_count]->_delta_q, calib_ric)){
                _ric[0] = calib_ric;
                para._Ric = calib_ric;
                _estimate_extrinsic = 1;
            }
        }
    }

    if (_solver_flag == INITIAL){
        // 只有滑窗中的KeyFrame数量达到指定大小的时候才开始优化.
        if (_frame_count == svar.GetInt("window_size", 20)){
            bool result = false;
            if (_estimate_extrinsic != 2 && (header - _initial_timestamp) > 0.1){
                result = InitialStructure();
                _initial_timestamp = header;
            }
            if (result){

            }
        }
        else
            _frame_count++;
    }
    else{

    }
}

bool Estimator::InitialStructure()
{
    // 检测Imu的可观性
    {
        std::map<double, ImageFrame>::iterator frame_it;
        Eigen::Vector3d sum_g;
        // 没有取第一帧
        for (frame_it = _all_image_frame.begin(), frame_it++; frame_it != _all_image_frame.end(); ++frame_it)
        {
            double sum_dt = frame_it->second._pre_integration->_sum_dt;
            Eigen::Vector3d tmp_g = frame_it->second._pre_integration->_delta_v / sum_dt;
            sum_g += tmp_g;
        }
        Eigen::Vector3d aver_g = sum_g * 1.0 / (_all_image_frame.size() - 1);

        double var = 0;
        for (frame_it = _all_image_frame.begin(), frame_it++; frame_it != _all_image_frame.end(); ++frame_it)
        {
            double sum_dt = frame_it->second._pre_integration->_sum_dt;
            Eigen::Vector3d tmp_g = frame_it->second._pre_integration->_delta_v / sum_dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
        }
        var = sqrt(var / _all_image_frame.size() - 1);

        if(var < 0.25)
            LOG(ERROR) << "Imu excitation not enough!";
    }

    // 遍历滑窗内所有的Features,以vector的形式保存滑窗内的所有特征点.
    std::vector<SfMFeature> sfm_feature;
    for(auto &it_per_id: _feature_manager._feature){
        int imu_j = it_per_id._start_frame - 1;
        SfMFeature tmp_feature;
        tmp_feature._state = false;
        tmp_feature._id = it_per_id._feature_id;
        for(auto &it_per_frame: it_per_id._feature_per_frame){
            imu_j++;
            Eigen::Vector3d pts_j = it_per_frame._point;
            // 这个_observation包含了所有观测到这个特征点的帧
            tmp_feature._observation.emplace_back(std::make_pair(imu_j, Eigen::Vector2d(pts_j.x(), pts_j.y())));
        }
        sfm_feature.emplace_back(tmp_feature);
    }

    Eigen::Matrix3d relative_R;
    Eigen::Vector3d relative_T;
    int l;
    if (!RelativePose(relative_R, relative_T, l)){
        return false;
    }

    InitialSfM sfm;
    // _frame_count+1代表传入的帧的数量
    std::vector<Eigen::Quaterniond> Q(_frame_count + 1);
    std::vector<Eigen::Vector3d> T(_frame_count + 1);
    std::map<int, Eigen::Vector3d> sfm_tracked_points;
    // 三角化和BA生成点云,相机到第l帧的位姿
    if (!sfm.Construct(_frame_count + 1, Q, T, l, relative_R, relative_T, sfm_feature, sfm_tracked_points)){
        _marginalization_flag = MARGIN_OLD;
        return false;
    }

    // 初始化成功了进入下面的步骤
    std::map<int, Eigen::Vector3d>::iterator it;
    std::map<double, ImageFrame>::iterator frame_it = _all_image_frame.begin();
    for(int i = 0; frame_it != _all_image_frame.end(); ++frame_it){
        if(frame_it->first == _headers[i]){
            frame_it->second._keyframe_flag = true;
            // 现在是第l帧到Imu的旋转
            frame_it->second._R = Q[i].toRotationMatrix() * _ric[0].transpose();
            frame_it->second._T = T[i];
            i++;
            continue;
        }
        if (frame_it->first > _headers[i])
            i++;

        // 只有没在滑动窗口中的帧会进入这一步.
        // 这都是之前初始化没成功的帧被margin了.
        // 后面的步骤就根据初始化完成的帧通过pnp获取之前的帧的位姿和跟踪到的特征点.
        // intial_r和initial_t 变成从第l帧到相机的位姿
        Eigen::Matrix3d initial_r = (Q[i].inverse()).toRotationMatrix();
        Eigen::Vector3d initial_t = -initial_r * T[i];
        cv::Mat rvec, t, tmp_r;
        cv::eigen2cv(initial_r, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(initial_t, t);
        frame_it->second._keyframe_flag = false;

        std::vector<cv::Point3f> pts_3_vector;
        std::vector<cv::Point2f> pts_2_vector;
        for(auto &id_pts: frame_it->second._points){
            int feature_id = id_pts.first;
            for(auto &i_p: id_pts.second){
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end()){
                    Eigen::Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.emplace_back(pts_3);

                    Eigen::Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.emplace_back(pts_2);
                }
            }
        }

        // FIXME: 感觉之前的帧如果不能跟踪到就一直不能跟踪了啊..这不就直接GG了吗
        if(pts_3_vector.size() < 6){
            std::cout << "pts_3_vector size: " << pts_3_vector.size() << std::endl;
            return false;
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat D;
        if(!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, true)){
            return false;
        }

        Eigen::MatrixXd pnp_r;
        Eigen::MatrixXd pnp_t;
        cv::Mat r;
        cv::Rodrigues(rvec, r);
        Eigen::MatrixXd tmp_pnp_r;
        cv::cv2eigen(r, tmp_pnp_r);
        pnp_r = tmp_pnp_r.transpose();
        cv::cv2eigen(t, pnp_t);
        pnp_t = pnp_r * (-pnp_t);

        // 变成Imu到相机的位姿了.
        frame_it->second._R = pnp_r * _ric[0].transpose();
        frame_it->second._T = pnp_t;
    }

    if(VisualInitialAlign())
        return true;
    else
        return false;
}

// FIXME: 感觉这个初始方法怪怪的,有优化的空间.
// 在滑窗中寻找与最新的关键帧共视关系较强的关键帧
bool Estimator::RelativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l)
{
    for(int i = 0; i < svar.GetInt("window_size", 20); ++i){
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres = _feature_manager.GetCorresponding(i, svar.GetInt("window_size", 10));
        if(corres.size() > 20){
            // 求取匹配的特征点在图像上的视差和(归一化平面上)
            double sum_parallax = 0;
            for (int j = 0; j < int(corres.size()); j++) {
                Eigen::Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Eigen::Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            // 求取平均视差
            double average_parallax = sum_parallax / corres.size();
            // 平均视差要大于一定阈值,并且能够有效地求解出变换矩阵.
            if (average_parallax * 460 > 30 && _motion_estimator.SolveRelativeRT(corres, relative_R, relative_T)){
                l = i;
                return true;
            }
        }
    }
    return false;
}

bool Estimator::VisualInitialAlign()
{
    Eigen::VectorXd x;
    // 要注意这个地方求解出的_g是在C0坐标系下的(也就是第l帧坐标系下).
    bool result = VisualImuAlignment(_all_image_frame, _Bgs, _g, x);
    if (!result){
        LOG(ERROR) << "solve g failed!";
        return false;
    }

    // 要注意到之前Construct函数成功后将那个时候的滑动窗口中的帧的_keyframe_flag标为true
    // 这里标记的帧可能和那个时候的帧不是一样的帧
    // FIXME: 感觉有些不对.
    for(int i = 0; i <= _frame_count; ++i){
        Eigen::Matrix3d Ri = _all_image_frame[_headers[i]]._R;
        Eigen::Vector3d Pi = _all_image_frame[_headers[i]]._T;
        _Rs[i] = Ri;
        _Ps[i] = Pi;
        _all_image_frame[_headers[i]]._keyframe_flag = true;
    }

    // 这个时候的逆深度为1 / -1.0;
    Eigen::VectorXd dep = _feature_manager.GetDepthVector();
    for(int i = 0; i < dep.size(); ++i)
        dep[i] = -1;
    _feature_manager.ClearDepth(dep);

    std::vector<Eigen::Vector3d> TIC_TMP(svar.GetInt("number_of_camera", 1));
    for(int i = 0; i < TIC_TMP.size(); ++i)
        TIC_TMP[i].setZero();
    _ric[0] = para._Ric;
    _feature_manager.SetRic(_ric);
    // FIXME: 感觉这里的写法有问题.
    // 并且这里没把Tic传入进去
    _feature_manager.Triangulate(_Ps, TIC_TMP, _ric);

    double s = (x.tail<1>())(0);
    // 优化了速度后需要重新传播.
    for(int i = 0; i <= svar.GetInt("window_size", 20); ++i)
        _pre_integrations[i]->Repropagate(Eigen::Vector3d::Zero(), _Bgs[i]);
    // 这里好像是把第一帧作为坐标0点.
    for(int i = _frame_count; i >= 0; --i)
        _Ps[i] = s * _Ps[i] - _Rs[i] * para._Tic[0] - (s * _Ps[0] - _Rs[0] * para._Tic[0]);
}

