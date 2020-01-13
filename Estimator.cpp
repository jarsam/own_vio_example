//
// Created by liu on 19-12-5.
//

#include "PoseLocalParameterization.h"
#include "Estimator.h"

void Estimator::SetParameter()
{
    for(int i = 0; i < svar.GetInt("camera_number", 1); ++i){
        _tic[i] = para._Tic;
        _ric[i] = para._Ric;
    }

    _feature_manager.SetRic(_ric);
    _td = 0;
}

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
    // 注意, 这里只加入一帧Imu信息, 而一帧图像会有很多Imu信息.
    // 也就是说这里只有一个_pre_integrations[0]只有一个Imu信息, 且不会进行预积分过程.
    if(!_pre_integrations[_frame_count])
        _pre_integrations[_frame_count] = new IntegrationBase(_acc0, _gyr0, _Bas[_frame_count], _Bgs[_frame_count]);
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
//        LOG(ERROR) << "_gyr0: " << _gyr0;
//        LOG(ERROR) << "angular_velocity: " << angular_velocity;
//        LOG(ERROR) << "Bgs: " << _Bgs[j];
//        LOG(ERROR) << "dt: " << dt;
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
    // 第一帧, 也就是frame_count=0时候的帧的_pre_integration 是NULL的
    // 之后的每一帧都是当前帧预积分好的这一帧内的旋转, 位移和速度变化.
    _all_image_frame.insert(std::make_pair(header, image_frame));

    // FIXME: 但是这个_tmp_pre_integration好像没有传入image_frame中.
    // 这个时候的初始化的意思就是开始下一帧的预积分了.
    // 这一帧的最后一个Imu信息作为初始值.
    _tmp_pre_integration = new IntegrationBase(_acc0, _gyr0, _Bas[_frame_count], _Bgs[_frame_count]);

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
                SolveOdometry();
                SlideWindow();
                _feature_manager.RemoveFailures();

                _lastR = _Rs[svar.GetInt("window_size")];
                _lastP = _Ps[svar.GetInt("window_size")];
                _lastR0 = _Rs[0];
                _lastP0 = _Ps[0];
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

        if (FailureDetection()){
            _failure_occur = true;
            ClearState();
            SetParameter();
            return;
        }

        SlideWindow();
        _feature_manager.RemoveFailures();
        _key_poses.clear();
        for(int i = 0; i <= svar.GetInt("window_size"); ++i)
            _key_poses.emplace_back(_Ps[i]);

        _lastR = _Rs[svar.GetInt("window_size")];
        _lastP = _Ps[svar.GetInt("window_size")];
        _lastR0 = _Rs[0];
        _lastP0 = _Ps[0];
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
    // 这个函数中做的工作就是在滑动窗口中找出一帧和最后一帧有一定视差的关键帧
    if (!RelativePose(relative_R, relative_T, l)){
        LOG(INFO) << "Not enough features or parallax; Move device around.";
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
            frame_it->second._R = Q[i].toRotationMatrix() * para._Ric.transpose();
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
        // 这个情况下失败的话就会一直失败, 但是这种失败条件很难
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
        frame_it->second._R = pnp_r * para._Ric.transpose();
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
        for(int i = 0; i < _Bgs.size(); ++i)
            _Bgs[i].setZero();
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
    // 这里的意思好像是计算出当前世界坐标系相当于水平坐标系的旋转, 也就是第一个关键帧相当于水平坐标系的旋转.
    // 这样就把所有关键帧的位置, 姿态和速度都转到了水平坐标系上了.

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

            delete _pre_integrations[svar.GetInt("window_size")];
            _pre_integrations[svar.GetInt("window_size", 20)] = new IntegrationBase{_acc0, _gyr0,
                                                                                    _Bas[svar.GetInt("window_size", 20)],
                                                                                    _Bgs[svar.GetInt("window_size", 20)]};
            _dt_buf[svar.GetInt("window_size", 20)].clear();
            _linear_acceleration_buf[svar.GetInt("window_size", 20)].clear();
            _angular_velocity_buf[svar.GetInt("window_size", 20)].clear();

            // FIXME: 这个是什么操作?
            // 这样的话不是_solver_flag的判断都没用了?
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

                _pre_integrations[_frame_count - 1]->PushBack(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);
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

            delete _pre_integrations[svar.GetInt("window_size")];
            _pre_integrations[svar.GetInt("window_size", 20)] = new IntegrationBase{_acc0, _gyr0,
                                                                                    _Bas[svar.GetInt("window_size", 20)],
                                                                                    _Bgs[svar.GetInt("window_size", 20)]};
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
        if (svar.GetInt("BA_ceres", 1))
            BackendOptimizationCeres();
        else
            BackendOptimizationEigen();
    }
}

void Estimator::BackendOptimizationCeres()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
    // 第一步: 添加待优化状态量
    // 添加[p, q](7), [speed, ba, bg](9)
    for(int i = 0; i < svar.GetInt("window_size", 20) + 1; ++i){
        // FIXME: 看不懂这个PoseLocalParameterization中的函数是干吗的.
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(_para_pose[i], POSE_SIZE, local_parameterization);
        problem.AddParameterBlock(_para_speed_bias[i], SPEED_BIAS);
    }

    for(int i = 0; i < svar.GetInt("camera_number", 1); ++i){
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(_para_ex_pose[i], POSE_SIZE, local_parameterization);
        if(!_estimate_extrinsic){
            LOG(INFO) << "Fix Extinsic Parameters";
            problem.SetParameterBlockConstant(_para_ex_pose[i]);
        }
        else
            LOG(INFO) << "Estimate Extinsic Parameters";
    }

    if (svar.GetInt("estimate_td", 1)){
        problem.AddParameterBlock(_para_td[0], 1);
    }

    Vector2Double();

    // 上一次边缘化的信息
    if (_last_marginalization_info){
        auto *marginalization_factor = new MarginalizationFactor(_last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL, _last_marginalization_parameter_blocks);
    }

    // FIXME: 这里的意思是某个滑动窗口时间太长了, 但是这在实际的SLAM中是有可能的
    // 比如无人机一直待在某个地方不动, 一直Margin Second New
    // 这里是从第1个窗口中直接开始的, 不应该是从第0个窗口开始吗?
    // 也就是说这里放入的窗口是1-20 而不是0-19.

    // 添加Imu的residual
    for(int i = 0; i < svar.GetInt("window_size", 20); ++i){
        int j = i + 1;
        if (_pre_integrations[j]->_sum_dt > 10.0)
            continue;
        auto* imu_factor = new ImuFactor(_pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, _para_pose[i], _para_speed_bias[i],
            _para_pose[j], _para_speed_bias[j]);
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
                auto* function_td = new ProjectionTdFactor(pts_i, pts_j,
                    it_per_id._feature_per_frame[0]._velocity, it_per_frame._velocity,
                    it_per_id._feature_per_frame[0]._cur_td, it_per_frame._cur_td,
                    it_per_id._feature_per_frame[0]._uv.y(), it_per_frame._uv.y());
                problem.AddResidualBlock(function_td, loss_function, _para_pose[imu_i], _para_pose[imu_j],
                    _para_ex_pose[0], _para_feature[feature_index], _para_td[0]);
            }
            else{
                auto* function_td = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(function_td, loss_function, _para_pose[imu_i], _para_pose[imu_j],
                                         _para_ex_pose[0], _para_feature[feature_index]);

            }
            feature_measurement_cnt++;
        }
    }

    double solver_time = svar.GetDouble("solver_time", 0.04);
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 100;
    if (_marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = solver_time * 4 / 5.0;
    else
        options.max_solver_time_in_seconds = solver_time;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    Double2Vector();

    if (_marginalization_flag == MARGIN_OLD){
        auto* marginalization_info = new MarginalizationInfo();
        Vector2Double();

        // 先验误差会一直保存, 而不是只使用一次
        // 如果上一次边缘化的信息存在
        // 要边缘化的参数块为para_pose[0], para_speed_bias[0], para_feature[feature_index](滑窗内的第feature_index个点的逆深度)
        if (_last_marginalization_info){
            std::vector<int> drop_set;
            for(int i = 0; i < _last_marginalization_parameter_blocks.size(); ++i){
                // 如果之前的先验误差中包含要边缘化的参数, 则这一次也要把这个标记上
                if (_last_marginalization_parameter_blocks[i] == _para_pose[0]
                    || _last_marginalization_parameter_blocks[i] == _para_speed_bias[0])
                    drop_set.emplace_back(i);
            }

            // 将上一次边缘化的参数块加入到边缘化factor中
            auto* marginalization_factor = new MarginalizationFactor(_last_marginalization_info);
            auto* residual_block_info = new ResidualBlockInfo(marginalization_factor,
                                                              nullptr, _last_marginalization_parameter_blocks,
                                                              drop_set);
            marginalization_info->AddResidualBlockInfo(residual_block_info);
        }
        // 添加Imu的先验, 只包含旧帧的Imu测量残差
        {
            if (_pre_integrations[1]->_sum_dt < 10.0){
                auto* imu_factor = new ImuFactor(_pre_integrations[1]);
                auto* residual_block_info = new ResidualBlockInfo(imu_factor, nullptr,
                    std::vector<double *> {_para_pose[0], _para_speed_bias[0],
                                           _para_pose[1], _para_speed_bias[1]}, std::vector<int>{0, 1});
                marginalization_info->AddResidualBlockInfo(residual_block_info);
            }
        }

        // 添加视觉的先验
        // 挑选出第一次观测帧为滑动窗口第一帧的路标点, 将对应的多组视觉观测放到marginlization_info中
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

                // 如果Imu的第一帧不是滑动窗口的第一帧, 则继续遍历下一帧
                if (imu_i != 0)
                    continue;
                // 该feature在滑动窗口的第一帧的归一化坐标
                Eigen::Vector3d pts_i = it_per_id._feature_per_frame[0]._point;

                for(auto &it_per_frame: it_per_id._feature_per_frame){
                    imu_j++;
                    if(imu_i == imu_j)
                        continue;

                    Eigen::Vector3d pts_j = it_per_frame._point;
                    if(svar.GetInt("estimate_td", 1)) {
                        auto* function_td =
                            new ProjectionTdFactor(pts_i, pts_j,
                                it_per_id._feature_per_frame[0]._velocity, it_per_frame._velocity,
                                it_per_id._feature_per_frame[0]._cur_td, it_per_frame._cur_td,
                                it_per_id._feature_per_frame[0]._uv.y(), it_per_frame._uv.y());

//                        problem.AddResidualBlock(function_td, loss_function, _para_pose[imu_i], _para_pose[imu_j],
//                            _para_ex_pose[0], _para_feature[feature_index], _para_td[0]);
                        auto* residual_block_info =
                            new ResidualBlockInfo(function_td, nullptr, std::vector<double*>{_para_pose[imu_i],
                                                                                             _para_pose[imu_j],
                                                                                             _para_ex_pose[0],
                                                                                             _para_feature[feature_index],
                                                                                             _para_td[0]}, std::vector<int>{0, 3});

                        marginalization_info->AddResidualBlockInfo(residual_block_info);
                    }
                    else{
                        auto* function_td = new ProjectionFactor(pts_i, pts_j);
//                        problem.AddResidualBlock(function_td, loss_function, _para_pose[imu_i], _para_pose[imu_j],
//                            _para_ex_pose[0], _para_feature[feature_index]);
                        auto* residual_block_info = new ResidualBlockInfo(function_td, nullptr,
                            std::vector<double*>{_para_pose[imu_i], _para_pose[imu_j], _para_ex_pose[0],
                                                 _para_feature[feature_index]}, std::vector<int>{0, 3});

                        marginalization_info->AddResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        marginalization_info->PreMarginalize();
        marginalization_info->Marginalize();

        // FIXME: 为什么是向右移位, 并且没保留逆深度的状态量
        // 这里让第i位指向i-1位就意味着抛弃了第0个的状态量.
        // 因为在下一次的优化中, 首先会调用Vector2Double()函数, 这样_para_pose[0]就会指向_para_pose[1]
        std::unordered_map<long, double * > addr_shift;
        for(int i = 1; i <= svar.GetInt("window_size", 20); ++i){
            addr_shift[reinterpret_cast<long>(_para_pose[i])] = _para_pose[i - 1];
            addr_shift[reinterpret_cast<long>(_para_speed_bias[i])] = _para_speed_bias[i - 1];
        }
        for(int i = 0; i < svar.GetInt("camera_number", 1); ++i)
            addr_shift[reinterpret_cast<long>(_para_ex_pose[i])] = _para_ex_pose[i];
        if(svar.GetInt("estimate_td", 1))
            addr_shift[reinterpret_cast<long>(_para_td[0])] = _para_td[0];

        std::vector<double *> parameter_blocks = marginalization_info->GetParameterBlocks(addr_shift);

        if(_last_marginalization_info)
            delete _last_marginalization_info;
        _last_marginalization_info = marginalization_info;
        _last_marginalization_parameter_blocks = parameter_blocks;
    }
    // 边缘化倒数第二帧
    else{
        // 只有当上一次margin了且上一次的参数中包含倒数第二帧的位姿信息才边缘化倒数第二帧
        if(_last_marginalization_info &&
            std::count(std::begin(_last_marginalization_parameter_blocks),
                std::end(_last_marginalization_parameter_blocks),
                _para_pose[svar.GetInt("window_size", 20) - 1])){
            auto* marginalization_info = new MarginalizationInfo();
            Vector2Double();

            std::vector<int> drop_set;
            for(int i = 0; i < _last_marginalization_parameter_blocks.size(); ++i){
                // 寻找倒数第二帧的位姿
                if(_last_marginalization_parameter_blocks[i] == _para_pose[svar.GetInt("window_size", 20) - 1])
                    drop_set.emplace_back(i);
            }

            auto* marginalization_factor = new MarginalizationFactor(_last_marginalization_info);
            auto* residual_block_info = new ResidualBlockInfo(marginalization_factor,
                                                              NULL, _last_marginalization_parameter_blocks, drop_set);
            marginalization_info->AddResidualBlockInfo(residual_block_info);

            marginalization_info->PreMarginalize();
            marginalization_info->Marginalize();

            std::unordered_map<long, double *> addr_shift;
            for(int i = 0; i <= svar.GetInt("window_size", 20); ++i){
                if (i == svar.GetInt("window_size", 20) - 1)
                    continue;
                else if (i == svar.GetInt("window_size", 20)){
                    addr_shift[reinterpret_cast<long>(_para_pose[i])] = _para_pose[i - 1];
                    addr_shift[reinterpret_cast<long>(_para_speed_bias[i])] = _para_speed_bias[i - 1];
                }
                else{
                    addr_shift[reinterpret_cast<long>(_para_pose[i])] = _para_pose[i];
                    addr_shift[reinterpret_cast<long>(_para_speed_bias[i])] = _para_speed_bias[i];
                }
            }
            for(int i = 0; i < svar.GetInt("camera_number", 1); ++i)
                addr_shift[reinterpret_cast<long>(_para_ex_pose[i])] = _para_ex_pose[i];
            if(svar.GetInt("estimate_td", 1))
                addr_shift[reinterpret_cast<long>(_para_td[0])] = _para_td[0];

            std::vector<double *> parameter_blocks = marginalization_info->GetParameterBlocks(addr_shift);

            delete _last_marginalization_info;
            _last_marginalization_info = marginalization_info;
            _last_marginalization_parameter_blocks = parameter_blocks;
        }
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
//        std::cout << "pose: " << _para_pose[i][0] << " " << _para_pose[i][1] << " " << _para_pose[i][2] << " " << _para_pose[i][3]
//                  <<" " << _para_pose[i][4] << " " << _para_pose[i][5] << " " << _para_pose[i][6] << std::endl;
//        std::cout << "bias: " << _para_speed_bias[i][0] << " " << _para_speed_bias[i][1] << " " << _para_speed_bias[i][2] << " " << _para_speed_bias[i][3]
//                  <<" " << _para_speed_bias[i][4] << " " << _para_speed_bias[i][5] << " " << _para_speed_bias[i][6] << " " << _para_speed_bias[i][7]
//                  << " " << _para_speed_bias[i][8] << std::endl;
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
//        std::cout << "ex pose: " << _para_ex_pose[i][0] << " " << _para_ex_pose[i][1] << " " << _para_ex_pose[i][2] << " " <<
//            _para_ex_pose[i][3] << " " << _para_ex_pose[i][4] << " " << _para_ex_pose[i][5] << " " << _para_ex_pose[i][6] << std::endl;
    }
    Eigen::VectorXd dep = _feature_manager.GetDepthVector();
    // FIXME: 这样不会越界吗? 还是滑动窗口中的特征点数量太少了. 初始化只有1000个点
    for(int i = 0; i < _feature_manager.GetFeatureCount(); ++i){
        _para_feature[i][0] = dep(i);
//        std::cout << "depth: " << _para_feature[i][0] << std::endl;
    }
    if (svar.GetInt("estiamte_td", 1)) {
        _para_td[0][0] = _td;
//        std::cout << "td: " << _para_td[0][0] << std::endl;
    }
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
    Eigen::Vector3d origin_R00 = Utility::R2YPR(Eigen::Quaterniond(_para_pose[0][6], _para_pose[0][3],
        _para_pose[0][4], _para_pose[0][5]).toRotationMatrix());

    // FIXME: 这应该是考虑到陀螺仪在roll轴和pitch轴上精度高吧.
    // rot_diff是根据滑窗中第一帧在优化前后的yaw偏差计算得到的旋转矩阵, 之后对滑窗内的所有帧都进行rot_diff的校正.
    // 这是因为在后端优化时, 我们并没有固定住第一帧的位姿不变, 而是将其作为优化变量进行调整.
    // 但是因为相机的yaw是不可观测的, 也就是说对于任意的yaw都满足优化函数, 所以优化后我们将偏航角旋转至优化之前的状态.
    double y_diff = origin_R0.x() - origin_R00.x();
    Eigen::Matrix3d rot_diff = Utility::YPR2R(Eigen::Vector3d(y_diff, 0, 0));
    if(abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0){
        rot_diff = _Rs[0] * Eigen::Quaterniond(_para_pose[0][6], _para_pose[0][3], _para_pose[0][4],
            _para_pose[0][5]).toRotationMatrix().transpose();
    }
    for(int i = 0; i <= svar.GetInt("window_size", 20); ++i){
        _Rs[i] = rot_diff * Eigen::Quaterniond(_para_pose[i][6], _para_pose[i][3], _para_pose[i][4], _para_pose[i][5]).normalized().toRotationMatrix();
        _Ps[i] = rot_diff * Eigen::Vector3d(_para_pose[i][0] - _para_pose[0][0], _para_pose[i][1] - _para_pose[0][1],
                                            _para_pose[i][2] - _para_pose[0][2]) + origin_P0;
        _Vs[i] = rot_diff * Eigen::Vector3d(_para_speed_bias[i][0], _para_speed_bias[i][1], _para_speed_bias[i][2]);
        _Bas[i] = Eigen::Vector3d(_para_speed_bias[i][3], _para_speed_bias[i][4], _para_speed_bias[i][5]);
        _Bgs[i] = Eigen::Vector3d(_para_speed_bias[i][6], _para_speed_bias[i][7], _para_speed_bias[i][8]);
    }

    for(int i = 0; i < svar.GetInt("camera_number", 1); ++i){
        _tic[i] = Eigen::Vector3d(_para_ex_pose[i][0], _para_ex_pose[i][1], _para_ex_pose[i][2]);
        _ric[i] = Eigen::Quaterniond(_para_ex_pose[i][6], _para_ex_pose[i][3], _para_ex_pose[i][4], _para_ex_pose[i][5]).toRotationMatrix();
    }

    Eigen::VectorXd dep = _feature_manager.GetDepthVector();
    for(int i = 0; i < _feature_manager.GetFeatureCount(); ++i)
        dep(i) = _para_feature[i][0];
    _feature_manager.SetDepth(dep);
    if (svar.GetInt("estimate_td"))
        _td = _para_td[0][0];
}

bool Estimator::FailureDetection()
{
    if (_feature_manager._last_track_num < 2){
        LOG(INFO) << "little feature: " << _feature_manager._last_track_num;
        return true;
    }
    if(_Bas[svar.GetInt("window_size")].norm() > 2.5){
        LOG(INFO) << "big IMU acc bias estimation: " << _Bas[svar.GetInt("window_size")].norm();
        return true;
    }
    if(_Bgs[svar.GetInt("window_size")].norm() > 1.0){
        LOG(INFO) << "big IMU gyr bias estimation: " << _Bgs[svar.GetInt("window_size")].norm();
        return true;
    }

    Eigen::Vector3d tmp_P = _Ps[svar.GetInt("window_size")];
    if ((tmp_P - _lastP).norm() > 5){
        LOG(INFO) << "big translation";
        return true;
    }
    if (abs(tmp_P.z() - _lastP.z()) > 1){
        LOG(INFO) << "big z translation";
        return true;
    }

    Eigen::Matrix3d tmp_R = _Rs[svar.GetInt("window_size")];
    Eigen::Matrix3d delta_R = tmp_R.transpose() * _lastR;
    Eigen::Quaterniond delta_Q(delta_R);
    // FIXME: 为什么要乘2
    double delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50){
        LOG(INFO) << "big delta_angle";
        return true;
    }

    return false;
}

void Estimator::ProblemSolve()
{
    LossFunction *loss_function = new CauthyLoss(1.0);
    Problem problem(Problem::SLAM_PROBLEM);
    std::vector<std::shared_ptr<VertexPose> > vertex_cam_vec;
    std::vector<std::shared_ptr<VertexSpeedBias> > vertex_speedbias_vec;
    int pose_dim = 0;

    // 先把外参数节点加入图优化, 这个节点在以后一直会被用到, 所以放在第一个.
    std::shared_ptr<VertexPose> vertex_ext(new VertexPose());
    {
        Eigen::VectorXd pose(7);
        pose << _para_ex_pose[0][0], _para_ex_pose[0][1], _para_ex_pose[0][2], _para_ex_pose[0][3],
            _para_ex_pose[0][4], _para_ex_pose[0][5], _para_ex_pose[0][6];
        vertex_ext->SetParameters(pose);

        if (!svar.GetInt("estimate_extrinsic", 1)){
            vertex_ext->SetFixed();
        }
        problem.AddVertex(vertex_ext);
        pose_dim += vertex_ext->LocalDimension();
    }

    for(int i = 0; i < svar.GetInt("window_size") + 1; ++i){
        std::shared_ptr<VertexPose> vertex_cam(new VertexPose());
        Eigen::VectorXd pose(7);
        pose << _para_pose[i][0], _para_pose[i][1], _para_pose[i][2], _para_pose[i][3],
            _para_pose[i][4], _para_pose[i][5], _para_pose[i][6];
        vertex_cam->SetParameters(pose);
        vertex_cam_vec.emplace_back(vertex_cam);
        problem.AddVertex(vertex_cam);
        pose_dim += vertex_cam->LocalDimension();

        std::shared_ptr<VertexSpeedBias> vertex_speedbias(new VertexSpeedBias());
        Eigen::VectorXd speedbias(9);
        speedbias << _para_speed_bias[i][0], _para_speed_bias[i][1], _para_speed_bias[i][2],
            _para_speed_bias[i][3], _para_speed_bias[i][4], _para_speed_bias[i][5],
            _para_speed_bias[i][6], _para_speed_bias[i][7], _para_speed_bias[i][8];
        vertex_speedbias->SetParameters(speedbias);
        vertex_speedbias_vec.emplace_back(vertex_speedbias);
        problem.AddVertex(vertex_speedbias);
        pose_dim += vertex_speedbias->LocalDimension();
    }

    for(int i = 0; i < svar.GetInt("window_size"); ++i){
        int j = i + 1;
        if (_pre_integrations[j]->_sum_dt > 10.0)
            continue;
        std::shared_ptr<EdgeImu> imu_edge(new EdgeImu(_pre_integrations[j]));
        std::vector<std::shared_ptr<Vertex> > edge_vertex;
        edge_vertex.emplace_back(vertex_cam_vec[i]);
        edge_vertex.emplace_back(vertex_speedbias_vec[i]);
        edge_vertex.emplace_back(vertex_cam_vec[j]);
        edge_vertex.emplace_back(vertex_speedbias_vec[j]);
        imu_edge->SetVertex(edge_vertex);
        problem.AddEdge(imu_edge);
    }

    std::vector<std::shared_ptr<VertexInverseDepth> > vertex_pt_vec;
    {
        int feature_index = -1;
        for(auto &it_per_id: _feature_manager._feature){
            it_per_id._used_num = it_per_id._feature_per_frame.size();
            if (!(it_per_id._used_num >= 2 && it_per_id._start_frame < svar.GetInt("window_size") - 2))
                continue;
            ++feature_index;
            int imu_i = it_per_id._start_frame, imu_j = imu_i - 1;
            Eigen::Vector3d pts_i = it_per_id._feature_per_frame[0]._point;

            std::shared_ptr<VertexInverseDepth> vertex_pt(new VertexInverseDepth());
            VecX inv_d(1);
            inv_d << _para_feature[feature_index][0];
            vertex_pt->SetParameters(inv_d);
            problem.AddVertex(vertex_pt);
            vertex_pt_vec.emplace_back(vertex_pt);

            for(auto &it_per_frame: it_per_id._feature_per_frame){
                imu_j++;
                if (imu_i == imu_j)
                    continue;
                Eigen::Vector3d pts_j = it_per_frame._point;
                std::shared_ptr<EdgeReprojection> edge(new EdgeReprojection(pts_i, pts_j));
                std::vector<std::shared_ptr<Vertex> > edge_vertex;
                edge_vertex.emplace_back(vertex_pt);
                edge_vertex.emplace_back(vertex_cam_vec[imu_i]);
                edge_vertex.emplace_back(vertex_cam_vec[imu_j]);
                edge_vertex.emplace_back(vertex_ext);

                edge->SetVertex(edge_vertex);
                edge->SetInformation(_project_sqrt_info.transpose() * _project_sqrt_info);
                edge->SetLossFunction(loss_function);
                problem.AddEdge(edge);
            }
        }
    }

    {
        // 已经有Prior了
        if(_H_prior.rows() > 0){
            problem.SetHessianPrior(_H_prior);
            problem.SetbPrior(_b_prior);
            problem.SetErrPrior(_err_prior);
            problem.SetJtPrior(_J_prior_inv);
            problem.ExtendHessiansPriorSize(15);// 扩展这个prior装新的pose
        }
    }

    problem.Solve(10);

    // update _b_prior
    if(_H_prior.rows() > 0){
        std::cout << "----------- update bprior -------------\n";
        std::cout << "             before: " << _b_prior.norm() << std::endl;
        std::cout << "                     " << _err_prior.norm() << std::endl;
        _b_prior = problem.GetbPrior();
        _err_prior = problem.GetErrPrior();
        std::cout << "             after: " << _b_prior.norm() << std::endl;
        std::cout << "                    " << _err_prior.norm() << std::endl;
    }

    for(int i = 0; i < svar.GetInt("window_size") + 1; ++i){
        VecX p = vertex_cam_vec[i]->Parameters();
        for(int j = 0; j < 7; ++j)
            _para_pose[i][j] = p[j];

        VecX speed_bias = vertex_speedbias_vec[i]->Parameters();
        for(int j = 0; j < 9; ++j)
            _para_speed_bias[i][j] = speed_bias[j];
    }

    for(int i = 0; i < vertex_pt_vec.size(); ++i){
        VecX f = vertex_pt_vec[i]->Parameters();
        _para_feature[i][0] = f[0];
    }
}

void Estimator::MarginOldFrame()
{
    LossFunction *loss_function = new CauthyLoss(1.0);
    Problem problem(Problem::SLAM_PROBLEM);
    std::vector<std::shared_ptr<VertexPose> > vertex_cam_vec;
    std::vector<std::shared_ptr<VertexSpeedBias> > vertex_speedbias_vec;
    int pose_dim = 0;

    // 先把外参数节点加入图优化, 这个节点在以后一直会被用到, 所以放在第一个.
    std::shared_ptr<VertexPose> vertex_ext(new VertexPose());
    {
        Eigen::VectorXd pose(7);
        pose << _para_ex_pose[0][0], _para_ex_pose[0][1], _para_ex_pose[0][2], _para_ex_pose[0][3],
            _para_ex_pose[0][4], _para_ex_pose[0][5], _para_ex_pose[0][6];
        vertex_ext->SetParameters(pose);

        if (!svar.GetInt("estimate_extrinsic", 1)){
            vertex_ext->SetFixed();
        }
        problem.AddVertex(vertex_ext);
        pose_dim += vertex_ext->LocalDimension();
    }

    for(int i = 0; i < svar.GetInt("window_size") + 1; ++i){
        std::shared_ptr<VertexPose> vertex_cam(new VertexPose());
        Eigen::VectorXd pose(7);
        pose << _para_pose[i][0], _para_pose[i][1], _para_pose[i][2], _para_pose[i][3],
            _para_pose[i][4], _para_pose[i][5], _para_pose[i][6];
        vertex_cam->SetParameters(pose);
        vertex_cam_vec.emplace_back(vertex_cam);
        problem.AddVertex(vertex_cam);
        pose_dim += vertex_cam->LocalDimension();

        std::shared_ptr<VertexSpeedBias> vertex_speedbias(new VertexSpeedBias());
        Eigen::VectorXd speedbias(9);
        speedbias << _para_speed_bias[i][0], _para_speed_bias[i][1], _para_speed_bias[i][2],
            _para_speed_bias[i][3], _para_speed_bias[i][4], _para_speed_bias[i][5],
            _para_speed_bias[i][6], _para_speed_bias[i][7], _para_speed_bias[i][8];
        vertex_speedbias->SetParameters(speedbias);
        vertex_speedbias_vec.emplace_back(vertex_speedbias);
        problem.AddVertex(vertex_speedbias);
        pose_dim += vertex_speedbias->LocalDimension();
    }

    if(_pre_integrations[1]->_sum_dt < 10.0){
        std::shared_ptr<EdgeImu> imu_edge(new EdgeImu(_pre_integrations[1]));
        std::vector<std::shared_ptr<Vertex> > edge_vertex;
        edge_vertex.emplace_back(vertex_cam_vec[0]);
        edge_vertex.emplace_back(vertex_speedbias_vec[0]);
        edge_vertex.emplace_back(vertex_cam_vec[1]);
        edge_vertex.emplace_back(vertex_speedbias_vec[1]);
        imu_edge->SetVertex(edge_vertex);
        problem.AddEdge(imu_edge);
    }

    {
        int feature_index = -1;
        for(auto &it_per_id: _feature_manager._feature){
            it_per_id._used_num = it_per_id._feature_per_frame.size();
            if (!(it_per_id._used_num >= 2 && it_per_id._start_frame < svar.GetInt("window_size") - 2))
                continue;
            ++feature_index;
            int imu_i = it_per_id._start_frame, imu_j = imu_i - 1;
            Eigen::Vector3d pts_i = it_per_id._feature_per_frame[0]._point;

            std::shared_ptr<VertexInverseDepth> vertex_pt(new VertexInverseDepth());
            VecX inv_d(1);
            inv_d << _para_feature[feature_index][0];
            vertex_pt->SetParameters(inv_d);
            problem.AddVertex(vertex_pt);

            for(auto &it_per_frame: it_per_id._feature_per_frame){
                imu_j++;
                if (imu_i == imu_j)
                    continue;
                Eigen::Vector3d pts_j = it_per_frame._point;
                std::shared_ptr<EdgeReprojection> edge(new EdgeReprojection(pts_i, pts_j));
                std::vector<std::shared_ptr<Vertex> > edge_vertex;
                edge_vertex.emplace_back(vertex_pt);
                edge_vertex.emplace_back(vertex_cam_vec[imu_i]);
                edge_vertex.emplace_back(vertex_cam_vec[imu_j]);
                edge_vertex.emplace_back(vertex_ext);

                edge->SetVertex(edge_vertex);
                edge->SetInformation(_project_sqrt_info.transpose() * _project_sqrt_info);
                edge->SetLossFunction(loss_function);
                problem.AddEdge(edge);
            }
        }
    }

    {
        if(_H_prior.rows() > 0){
            problem.SetHessianPrior(_H_prior);
            problem.SetbPrior(_b_prior);
            problem.SetErrPrior(_err_prior);
            problem.SetJtPrior(_J_prior_inv);
            problem.ExtendHessiansPriorSize(15);
        }
        else{
            _H_prior = MatXX(pose_dim, pose_dim);
            _H_prior.setZero();
            _b_prior = VecX(pose_dim);
            _b_prior.setZero();
            problem.SetHessianPrior(_H_prior);
            problem.SetbPrior(_b_prior);
        }
    }

    std::vector<std::shared_ptr<Vertex> > marg_vertex;
    marg_vertex.emplace_back(vertex_cam_vec[0]);
    marg_vertex.emplace_back(vertex_speedbias_vec[0]);
    problem.Marginalize(marg_vertex, pose_dim);
    _H_prior = problem.GetHessianPrior();
    _b_prior = problem.GetbPrior();
    _err_prior = problem.GetErrPrior();
    _J_prior_inv = problem.GetJtPrior();
}

void Estimator::MarginNewFrame()
{
    Problem problem(Problem::SLAM_PROBLEM);
    std::vector<std::shared_ptr<VertexPose> > vertex_cam_vec;
    std::vector<std::shared_ptr<VertexSpeedBias> > vertex_speedbias_vec;
    int pose_dim = 0;

    // 先把外参数节点加入图优化, 这个节点在以后一直会被用到, 所以放在第一个.
    std::shared_ptr<VertexPose> vertex_ext(new VertexPose());
    {
        Eigen::VectorXd pose(7);
        pose << _para_ex_pose[0][0], _para_ex_pose[0][1], _para_ex_pose[0][2], _para_ex_pose[0][3],
            _para_ex_pose[0][4], _para_ex_pose[0][5], _para_ex_pose[0][6];
        vertex_ext->SetParameters(pose);
        problem.AddVertex(vertex_ext);
        pose_dim += vertex_ext->LocalDimension();
    }

    for(int i = 0; i < svar.GetInt("window_size") + 1; ++i){
        std::shared_ptr<VertexPose> vertex_cam(new VertexPose());
        Eigen::VectorXd pose(7);
        pose << _para_pose[i][0], _para_pose[i][1], _para_pose[i][2], _para_pose[i][3],
            _para_pose[i][4], _para_pose[i][5], _para_pose[i][6];
        vertex_cam->SetParameters(pose);
        vertex_cam_vec.emplace_back(vertex_cam);
        problem.AddVertex(vertex_cam);
        pose_dim += vertex_cam->LocalDimension();

        std::shared_ptr<VertexSpeedBias> vertex_speedbias(new VertexSpeedBias());
        Eigen::VectorXd speedbias(9);
        speedbias << _para_speed_bias[i][0], _para_speed_bias[i][1], _para_speed_bias[i][2],
            _para_speed_bias[i][3], _para_speed_bias[i][4], _para_speed_bias[i][5],
            _para_speed_bias[i][6], _para_speed_bias[i][7], _para_speed_bias[i][8];
        vertex_speedbias->SetParameters(speedbias);
        vertex_speedbias_vec.emplace_back(vertex_speedbias);
        problem.AddVertex(vertex_speedbias);
        pose_dim += vertex_speedbias->LocalDimension();
    }

    {
        if (_H_prior.rows() > 0){
            problem.SetHessianPrior(_H_prior);
            problem.SetbPrior(_b_prior);
            problem.SetErrPrior(_err_prior);
            problem.SetJtPrior(_J_prior_inv);
            problem.ExtendHessiansPriorSize(15);
        }
    }

    std::vector<std::shared_ptr<Vertex> > marg_vertex;
    marg_vertex.emplace_back(vertex_cam_vec[svar.GetInt("window_size") - 1]);
    marg_vertex.emplace_back(vertex_speedbias_vec[svar.GetInt("window_size") - 1]);
    problem.Marginalize(marg_vertex, pose_dim);
    _H_prior = problem.GetHessianPrior();
    _b_prior = problem.GetbPrior();
    _err_prior = problem.GetErrPrior();
    _J_prior_inv = problem.GetJtPrior();
}

void Estimator::BackendOptimizationEigen()
{
    _project_sqrt_info = (para._camera_intrinsics[0] + para._camera_intrinsics[1])/2/1.5 * Eigen::Matrix2d::Identity();
    Vector2Double();
    // 构建求解器
    ProblemSolve();

    Double2Vector();

    if(_marginalization_flag == MARGIN_OLD){
        Vector2Double();
        MarginOldFrame();
    }
    else{
        if(_H_prior.rows() > 0){
            Vector2Double();
            MarginNewFrame();
        }
    }
}
