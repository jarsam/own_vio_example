//
// Created by liu on 19-12-5.
//

#include "PoseLocalParameterization.h"
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
    {
        LOG(INFO) << "Margin Old";
        _marginalization_flag = MARGIN_OLD;// KeyFrame
    }
    else{
        LOG(INFO) << "Margin Second New";
        _marginalization_flag = MARGIN_SECOND_NEW; // Non-KeyFrame
    }

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
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres =
                _feature_manager.GetCorresponding(_frame_count - 1, _frame_count);
            Eigen::Matrix3d calib_ric;
            if (_initial_ex_rotation.CalibrationExRotation(corres, _pre_integrations[_frame_count]->_delta_q, calib_ric)){
                LOG(INFO) << "Calibration Successed";
                _ric[0] = calib_ric;
                para._Ric = calib_ric;
                _estimate_extrinsic = 1;
            }
            else
                LOG(INFO) << "Calibration Failed";
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
                LOG(INFO) << "Initial Construct Successed.";
                _solver_flag = NON_LINEAR;
            }
            else {
                LOG(INFO) << "Initial Construct Failed";
                SlideWindow();
            }
        }
        else
            _frame_count++;
    }
    else{
        SolveOdometry();
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
        LOG(INFO) << "RelativePose Failed";
        return false;
    }

    InitialSfM sfm;
    // _frame_count+1代表传入的帧的数量
    std::vector<Eigen::Quaterniond> Q(_frame_count + 1);
    std::vector<Eigen::Vector3d> T(_frame_count + 1);
    std::map<int, Eigen::Vector3d> sfm_tracked_points;
    // 三角化和BA生成点云,相机到第l帧的位姿
    if (!sfm.Construct(_frame_count + 1, Q, T, l, relative_R, relative_T, sfm_feature, sfm_tracked_points)){
        LOG(INFO) << "sfm Construct Failed";
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
            LOG(INFO) << "SolvePnP Failed";
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
    {
        LOG(INFO) << "Visual Initial Align Failed";
        return false;
    }
}

// FIXME: 感觉这个初始方法怪怪的,有优化的空间.
// 在滑窗中寻找与最新的关键帧共视关系较强的关键帧
bool Estimator::RelativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l)
{
    for(int i = 0; i < svar.GetInt("window_size", 20); ++i){
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres = _feature_manager.GetCorresponding(i, svar.GetInt("window_size", 20));
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

    std::vector<Eigen::Vector3d> TIC_TMP(svar.GetInt("camera_number", 1));
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
        _Ps[i] = s * _Ps[i] - _Rs[i] * para._Tic - (s * _Ps[0] - _Rs[0] * para._Tic);
    int kv = -1;
    std::map<double, ImageFrame>::iterator frame_i;
    for(frame_i = _all_image_frame.begin(); frame_i != _all_image_frame.end(); ++frame_i){
        if(frame_i->second._keyframe_flag){
            kv++;
            // 将原本的Imu坐标系的值转到世界坐标系下
            // 将原本的Imu坐标系的值转到世界坐标系下
            _Vs[kv] = frame_i->second._R * x.segment<3>(kv * 3);
        }
    }
    for(auto &it_per_id: _feature_manager._feature){
        it_per_id._used_num = it_per_id._feature_per_frame.size();
        if(!(it_per_id._used_num >= 2 && it_per_id._start_frame < svar.GetInt("window_size", 20) - 2))
            continue;
        it_per_id._estimated_depth *= s;
    }

    // FIXME: 需要写完后继续看看, 这下面的代码的意思应该是将所有的值转到滑动窗口的第0帧的坐标系.
    // 以第0帧为初始帧,重力指向第0帧的{0, 0, 1}
    Eigen::Matrix3d R0 = Utility::G2R(_g);
    double yaw = Utility::R2YPR(R0 * _Rs[0]).x();
    R0 = Utility::YPR2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    _g = R0 * _g;
    Eigen::Matrix3d rot_diff = R0;
    for(int i = 0; i <= _frame_count; ++i){
        _Ps[i] = rot_diff * _Ps[i];
        _Rs[i] = rot_diff * _Rs[i];
        _Vs[i] = rot_diff * _Vs[i];
    }

    return true;
}

void Estimator::SlideWindow()
{
    if (_marginalization_flag == MARGIN_OLD){
        double t0 = _headers[0];
        _backR0 = _Rs[0];
        _backP0 = _Ps[0];
        if (_frame_count == svar.GetInt("window_size", 20)){
            for(int i = 0; i < svar.GetInt("window_size", 20); ++i){
                std::swap(_pre_integrations[i], _pre_integrations[i + 1]);
                _dt_buf[i].swap(_dt_buf[i + 1]);
                _linear_acceleration_buf[i].swap(_linear_acceleration_buf[i + 1]);
                _angular_velocity_buf[i].swap(_angular_velocity_buf[i + 1]);
                _headers[i] = _headers[i + 1];
                _Ps[i].swap(_Ps[i + 1]);
                _Vs[i].swap(_Vs[i + 1]);
                _Rs[i].swap(_Rs[i + 1]);
                _Bas[i].swap(_Bas[i + 1]);
                _Bgs[i].swap(_Bgs[i + 1]);
            }
            _headers[svar.GetInt("window_size", 20)] = _headers[svar.GetInt("window_size", 20) - 1];
            _Ps[svar.GetInt("window_size", 20)] = _Ps[svar.GetInt("window_size", 20) - 1];
            _Vs[svar.GetInt("window_size", 20)] = _Vs[svar.GetInt("window_size", 20) - 1];
            _Rs[svar.GetInt("window_size", 20)] = _Rs[svar.GetInt("window_size", 20) - 1];
            _Bas[svar.GetInt("window_size", 20)] = _Bas[svar.GetInt("window_size", 20) - 1];
            _Bgs[svar.GetInt("window_size", 20)] = _Bgs[svar.GetInt("window_size", 20) - 1];

            _pre_integrations[svar.GetInt("window_size", 20)] =
                std::shared_ptr<IntegrationBase>(new IntegrationBase{_acc0, _gyr0,
                                                                     _Bas[svar.GetInt("window_size", 20)],
                                                                     _Bgs[svar.GetInt("window_size", 20)]});
            _dt_buf[svar.GetInt("window_size", 20)].clear();
            _linear_acceleration_buf[svar.GetInt("window_size", 20)].clear();
            _angular_velocity_buf[svar.GetInt("window_size", 20)].clear();

            // FIXME: 这个是什么操作?
            if (true || _solver_flag == INITIAL){
                std::map<double, ImageFrame>::iterator it0;
                it0 = _all_image_frame.find(t0);
                it0->second._pre_integration = nullptr;

                // 将滑动窗口之前的帧去除了
                for(std::map<double, ImageFrame>::iterator it = _all_image_frame.begin(); it != it0; ++it){
                    it->second._pre_integration = nullptr;
                }
                _all_image_frame.erase(_all_image_frame.begin(), it0);
                _all_image_frame.erase(t0);
            }
            SlideWindowOld();
        }
    }
    else{
        // FIXME: 这不是margin了最新一帧吗?
        // 只有当滑动窗口满了才margin new帧
        // 将_frame_count帧margin了, 将这个帧的信息传入到上一帧中.
        if(_frame_count == svar.GetInt("window_size", 20)){
            for(unsigned int i = 0; i < _dt_buf[_frame_count].size(); ++i){
                double tmp_dt = _dt_buf[_frame_count][i];
                Eigen::Vector3d tmp_linear_acceleration = _linear_acceleration_buf[_frame_count][i];
                Eigen::Vector3d tmp_angular_velocity = _angular_velocity_buf[_frame_count][i];

                _pre_integrations[_frame_count - 1] ->PushBack(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);
                _dt_buf[_frame_count - 1].emplace_back(tmp_dt);
                _linear_acceleration_buf[_frame_count - 1].emplace_back(tmp_linear_acceleration);
                _angular_velocity_buf[_frame_count - 1].emplace_back(tmp_angular_velocity);
            }

            _headers[_frame_count - 1] = _headers[_frame_count];
            _Ps[_frame_count - 1] = _Ps[_frame_count];
            _Vs[_frame_count - 1] = _Vs[_frame_count];
            _Rs[_frame_count - 1] = _Rs[_frame_count];
            _Bas[_frame_count - 1] = _Bas[_frame_count];
            _Bgs[_frame_count - 1] = _Bgs[_frame_count];

            _pre_integrations[svar.GetInt("window_size", 20)] =
                std::shared_ptr<IntegrationBase>(new IntegrationBase{_acc0, _gyr0,
                                                                     _Bas[svar.GetInt("window_size", 20)],
                                                                     _Bgs[svar.GetInt("window_size", 20)]});
            _dt_buf[svar.GetInt("window_size", 20)].clear();
            _linear_acceleration_buf[svar.GetInt("window_size", 20)].clear();
            _angular_velocity_buf[svar.GetInt("window_size", 20)].clear();

            SlideWindowNew();
        }
    }
}

void Estimator::SlideWindowOld()
{
    _sum_of_back++;
    // 只有初始化成功后SlideWindow才需要考虑特征点深度
    bool shift_depth = _solver_flag == NON_LINEAR ? true : false;
    // 将特征点去除.
    if (shift_depth){
        Eigen::Matrix3d R0, R1;
        Eigen::Vector3d P0, P1;
        R0 = _backR0 * _ric[0];
        R1 = _Rs[0] * _ric[0];
        P0 = _backP0 + _backR0 * _tic[0];
        P1 = _Ps[0] + _Rs[0] * _tic[0];
        _feature_manager.RemoveBackShiftDepth(R0, P0, R1, P1);
    }
    else
        _feature_manager.RemoveBack();
}

void Estimator::SlideWindowNew()
{
    _sum_of_front++;
    _feature_manager.RemoveFront(_frame_count);
}

void Estimator::SolveOdometry()
{
    if (_frame_count < svar.GetInt("window_size", 20))
        return;
    if (_solver_flag == NON_LINEAR){
        // 三角化那些没有被三角化的点.
        _feature_manager.Triangulate(_Ps, _tic, _ric);
        BackendOptimization();
    }
}

void Estimator::BackendOptimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
    // 第一步: 添加待优化状态量
    // 添加[p, q](7), [speed, ba, bg](9)
    for(int i = 0; i < svar.GetInt("window_size", 20); ++i){
        // FIXME: 看不懂这个PoseLocalParameterization中的函数是干吗的.
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(_para_pose[i].data(), POSE_SIZE, local_parameterization);
        problem.AddParameterBlock(_para_speed_bias[i].data(), SPEED_BIAS);
    }

    for(int i = 0; i < svar.GetInt("camera_number", 1); ++i){
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(_para_ex_pose[i].data(), POSE_SIZE, local_parameterization);
        if(_estimate_extrinsic){
            LOG(INFO) << "Fix Extinsic Parameters";
            problem.SetParameterBlockConstant(_para_ex_pose[i].data());
        }
        else
            LOG(INFO) << "Estimate Extinsic Parameters";
    }

    if (svar.GetInt("estimate_td", 1)){
        problem.AddParameterBlock(_para_td[0].data(), 1);
    }

    Vector2Double();

    // 上一次边缘化的信息
    if (_last_marginalization_info){

    }
    // 添加Imu的residual
    for(int i = 0; i < svar.GetInt("window_size", 20); ++i){
        int j = i + 1;
        // FIXME: 这里的意思是某个滑动窗口时间太长了, 但是这在实际的SLAM中是有可能的
        // 比如无人机一直待在某个地方不动, 一直Margin Second New
        if (_pre_integrations[j]->_sum_dt > 10.0)
            continue;
        std::shared_ptr<ImuFactor> imu_factor = std::shared_ptr<ImuFactor>(new ImuFactor(_pre_integrations[j]));
        problem.AddResidualBlock(imu_factor.get(), NULL, _para_pose[i].data(), _para_speed_bias[i].data(), _para_pose[j].data(), _para_speed_bias[j].data());
    }

    // 特征点的测量值的数量
    int feature_measurement_cnt = 0;
    int feature_index = -1;
    for(auto &it_per_id: _feature_manager._feature){
        it_per_id._used_num = it_per_id._feature_per_frame.size();
        if (!(it_per_id._used_num >= 2 && it_per_id._start_frame < svar.GetInt("window_size", 20) - 2))
            continue;
        ++feature_index;
        //  得到观测到该特征点的首帧
        int imu_i = it_per_id._start_frame, imu_j = imu_i - 1;
        // 得到首帧观测到的特征点
        Eigen::Vector3d pts_i = it_per_id._feature_per_frame[0]._point;
        // it_per_frame 是除了首帧之外的点.
        for(auto &it_per_frame: it_per_id._feature_per_frame){
            imu_j++;
            if(imu_i == imu_j)
                continue;

            Eigen::Vector3d pts_j = it_per_frame._point;
            if(svar.GetInt("estimate_td", 1)){
                std::shared_ptr<ProjectionTdFactor> function_td =
                    std::shared_ptr<ProjectionTdFactor>(new ProjectionTdFactor(pts_i, pts_j,
                        it_per_id._feature_per_frame[0]._velocity, it_per_frame._velocity,
                        it_per_id._feature_per_frame[0]._cur_td, it_per_frame._cur_td,
                        it_per_id._feature_per_frame[0]._uv.y(), it_per_frame._uv.y()));
                problem.AddResidualBlock(function_td.get(), loss_function, _para_pose[imu_i].data(), _para_pose[imu_j].data(),
                                         _para_ex_pose[0].data(), _para_feature[feature_index].data(), _para_td[0].data());
            }
            else{
                std::shared_ptr<ProjectionFactor> function_td =
                    std::shared_ptr<ProjectionFactor>(new ProjectionFactor(pts_i, pts_j));
                problem.AddResidualBlock(function_td.get(), loss_function, _para_pose[imu_i].data(), _para_pose[imu_j].data(),
                                         _para_ex_pose[0].data(), _para_feature[feature_index].data());

            }
            feature_measurement_cnt++;
        }
    }

    if (_relocalization_info){
        std::shared_ptr<ceres::LocalParameterization> local_parameterization =
            std::shared_ptr<ceres::LocalParameterization>(new PoseLocalParameterization());
        problem.AddParameterBlock(_relo_pose.data(), POSE_SIZE, local_parameterization.get());
        int retrive_feature_index = 0;
        int feature_index = -1;
        for(auto &it_per_id: _feature_manager._feature){
            it_per_id._used_num = it_per_id._feature_per_frame.size();
            if (!(it_per_id._used_num >= 2 && it_per_id._start_frame < svar.GetInt("window_size", 20) - 2))
                continue;
            ++feature_index;
            int start = it_per_id._start_frame;
            if (start <= _relo_frame_local_index){
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 100;
    if (_marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = 0.04 * 4 / 5.0;
    else
        options.max_solver_time_in_seconds = 0.04;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    Double2Vector();

    if (_marginalization_flag == MARGIN_OLD){
        std::shared_ptr<MarginalizationInfo> marginalization_info(new MarginalizationInfo());
        Vector2Double();

        if (_last_marginalization_info){
            std::vector<int> drop_set;

        }
        // 添加Imu的先验, 只包含旧帧的Imu测量残差
        {
            if (_pre_integrations[1]->_sum_dt < 10.0){
                std::shared_ptr<ImuFactor> imu_factor(new ImuFactor(_pre_integrations[1]));
                std::shared_ptr<ResidualBlockInfo> residual_block_info(
                    new ResidualBlockInfo(imu_factor, nullptr,
                        std::vector<double *> {_para_pose[0], _para_speed_bias[0], _para_pose[1], _para_speed_bias[1]},
                        std::vector<int>{0, 1}));
                marginalization_info->AddResidualBlockInfo(residual_block_info);
            }
        }

        // 添加视觉的先验
        {
            int feature_index = -1;
            // 遍历滑窗内的所有features
            for(auto &it_per_id: _feature_manager._feature){
                // 该特征点被观测到的次数
                it_per_id._used_num = it_per_id._feature_per_frame.size();
                if(!(it_per_id._used_num >= 2 && it_per_id._start_frame < svar.GetInt("window_size", 20) - 2))
                    continue;
                ++feature_index;

                int imu_i = it_per_id._start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;
                // 该feature在起始帧的归一化坐标
                Eigen::Vector3d pts_i = it_per_id._feature_per_frame[0]._point;

                for(auto &it_per_frame: it_per_id._feature_per_frame){
                    imu_j++;
                    if(imu_i == imu_j)
                        continue;

                    Eigen::Vector3d pts_j = it_per_frame._point;
                    if(svar.GetInt("estimate_td", 1)) {
                        std::shared_ptr<ProjectionTdFactor> function_td =
                            std::shared_ptr<ProjectionTdFactor>(new ProjectionTdFactor(pts_i, pts_j,
                                it_per_id._feature_per_frame[0]._velocity, it_per_frame._velocity,
                                it_per_id._feature_per_frame[0]._cur_td, it_per_frame._cur_td,
                                it_per_id._feature_per_frame[0]._uv.y(), it_per_frame._uv.y()));

                        std::shared_ptr<ResidualBlockInfo> residual_block_info =
                            std::shared_ptr<ResidualBlockInfo>(new ResidualBlockInfo(function_td,
                                std::vector<double*>{_para_pose[imu_i], _para_pose[imu_j], _para_ex_pose[0],
                                                     _para_feature[feature_index], _para_td[0]}, std::vector<int>{0, 3}));

                        marginalization_info->AddResidualBlockInfo(residual_block_info);
                    }
                    else{
                        std::shared_ptr<ProjectionTdFactor> function_td =
                            std::shared_ptr<ProjectionTdFactor>(new ProjectionFactor(pts_i, pts_j));
                        std::shared_ptr<ResidualBlockInfo> residual_block_info =
                            std::shared_ptr<ResidualBlockInfo>(
                                new ResidualBlockInfo(function_td,
                                    std::vector<double*>{_para_pose[imu_i], _para_pose[imu_j], _para_ex_pose[0],
                                                         _para_feature[feature_index], std::vector<int>{0, 3}}));

                        marginalization_info->AddResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        marginalization_info->PreMarginalize();
        marginalization_info->Marginalize();
    }
}

void Estimator::Vector2Double()
{
    for(int i = 0; i <= svar.GetInt("window_size", 20); ++i){
        _para_pose[i][0] = _Ps[i].x();
        _para_pose[i][1] = _Ps[i].y();
        _para_pose[i][2] = _Ps[i].z();
        Eigen::Quaterniond q{_Rs[i]};
        _para_pose[i][3] = q.x();
        _para_pose[i][4] = q.y();
        _para_pose[i][5] = q.z();
        _para_pose[i][6] = q.w();

        _para_speed_bias[i][0] = _Vs[i].x();
        _para_speed_bias[i][1] = _Vs[i].y();
        _para_speed_bias[i][2] = _Vs[i].z();

        _para_speed_bias[i][3] = _Bas[i].x();
        _para_speed_bias[i][4] = _Bas[i].y();
        _para_speed_bias[i][5] = _Bas[i].z();

        _para_speed_bias[i][6] = _Bgs[i].x();
        _para_speed_bias[i][7] = _Bgs[i].y();
        _para_speed_bias[i][8] = _Bgs[i].z();
    }
    for(int i = 0; i < svar.GetInt("camera_number", 1); ++i){
        _para_ex_pose[i][0] = _tic[i].x();
        _para_ex_pose[i][1] = _tic[i].y();
        _para_ex_pose[i][2] = _tic[i].z();
        Eigen::Quaterniond q{_ric[i]};
        _para_ex_pose[i][3] = q.x();
        _para_ex_pose[i][4] = q.y();
        _para_ex_pose[i][5] = q.z();
        _para_ex_pose[i][6] = q.w();
    }
    Eigen::VectorXd dep = _feature_manager.GetDepthVector();
    // FIXME: 这样不会越界吗? 还是滑动窗口中的特征点数量太少了.
    for(int i = 0; i < _feature_manager.GetFeatureCount(); ++i)
        _para_feature[i][0] = dep(i);
    if (svar.GetInt("estiamte_td", 1))
        _para_td[0][0] = _td;
}

void Estimator::Double2Vector()
{
    Eigen::Vector3d origin_R0 = Utility::R2YPR(_Rs[0]);
    Eigen::Vector3d origin_P0 = _Ps[0];

    if (_failure_occur){
        origin_R0 = Utility::R2YPR(_lastR0);
        origin_P0 = _lastP0;
        _failure_occur = 0;
    }
}
