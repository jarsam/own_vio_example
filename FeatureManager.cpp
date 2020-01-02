//
// Created by liu on 19-12-6.
//

#include "FeatureManager.h"

FeatureManager::FeatureManager(std::vector<Eigen::Matrix3d> &Rs): _R(Rs.data())
{
    for(int i = 0; i < svar.GetInt("camera_number", 1); ++i)
        _ric.emplace_back(Eigen::Matrix3d::Identity());
}

void FeatureManager::SetRic(std::vector<Eigen::Matrix3d> &Ric)
{
    for(int i = 0; i < svar.GetInt("camera_number", 1); ++i)
        _ric[i] = Ric[i];
}

bool FeatureManager::AddFeatureCheckParallax(int frame_count,
                                             const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                                             double td)
{
    double parallax_sum = 0;
    int parallax_num = 0;
    _last_track_num = 0;
    for (auto &id_pts: image){
        FeaturePerFrame feature_per_frame(id_pts.second[0].second, td);
        int feature_id = id_pts.first;
        auto it = find_if(_feature.begin(), _feature.end(), [feature_id](const FeaturePerId &it){
            return it._feature_id == feature_id;
        });
        if (it == _feature.end()){
            _feature.emplace_back(FeaturePerId(feature_id, frame_count));
            _feature.back()._feature_per_frame.emplace_back(feature_per_frame);
        }
        else if (it->_feature_id == feature_id){
            it->_feature_per_frame.emplace_back(feature_per_frame);
            _last_track_num++;
        }
    }

    // 如果滑窗内的关键帧的个数小于2或者总共被跟踪到的次数小于20.
    // 这里的总共被跟踪的次数是指这一帧被跟踪到的已经存在的特征点数量.
    // 这样的话就MARGIN_OLD.
    if (frame_count < 2 || _last_track_num < 20)
        return true;

    // 计算共视关系,parallax_num为满足要求的Feature的个数.
    for (auto &it_per_id: _feature){
        // 至少有两帧观测到该特征点(不包括当前帧)
        // 然后比较观测到该Feature的倒数第二帧和倒数第三帧的视差
        if (it_per_id._start_frame <= frame_count - 2 &&
            it_per_id._start_frame + int(it_per_id._feature_per_frame.size()) - 1 >= frame_count - 1){
            parallax_sum += CompensatedParallax(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
        return true;
    // 平均视差要大于某个阈值,这里取10个像素点.
    else
        return parallax_sum / parallax_num >=
            2 * svar.GetDouble("min_parallax", 10) / ((para._camera_intrinsics[0] + para._camera_intrinsics[1]));
}

// 这个函数实际上是求取该特征点在两帧的归一化平面上的坐标点的距离
double FeatureManager::CompensatedParallax(const FeaturePerId &it_per_id, int frame_count)
{
    const FeaturePerFrame &frame_i = it_per_id._feature_per_frame[frame_count - 2 - it_per_id._start_frame];
    const FeaturePerFrame &frame_j = it_per_id._feature_per_frame[frame_count - 1 - it_per_id._start_frame];

    Eigen::Vector3d p_i = frame_i._point;
    Eigen::Vector3d p_j = frame_j._point;

    double u_j = p_j(0);
    double v_j = p_j(1);
    // FIXME: 感觉不需要归一化
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j;
    double dv = v_i - v_j;

    double ans = std::max(0.0, sqrt(du * du + dv * dv));

    return ans;
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> FeatureManager::GetCorresponding(int frame_count_l,
                                                                                          int frame_count_r)
{
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
    for(auto &it: _feature){
        if (it._start_frame <= frame_count_l && it.EndFrame() >= frame_count_r){
            Eigen::Vector3d a = Eigen::Vector3d::Zero(), b = Eigen::Vector3d::Zero();
            int idx_l = frame_count_l - it._start_frame;
            int idx_r = frame_count_r - it._start_frame;
            a = it._feature_per_frame[idx_l]._point;
            b = it._feature_per_frame[idx_r]._point;
            corres.emplace_back(std::make_pair(a, b));
        }
    }
    return corres;
}

// 读取特征点的逆深度
Eigen::VectorXd FeatureManager::GetDepthVector()
{
    Eigen::VectorXd dep_vec(GetFeatureCount());
    int feature_index = -1;
    for(auto &it_per_id: _feature){
        it_per_id._used_num = it_per_id._feature_per_frame.size();
        if (!(it_per_id._used_num >= 2 && it_per_id._start_frame < svar.GetInt("window_size", 20) - 2))
            continue;
        dep_vec(++feature_index) = 1. / it_per_id._estimated_depth;
    }
    return dep_vec;
}

int FeatureManager::GetFeatureCount()
{
    int cnt = 0;
    for(auto &it: _feature){
        it._used_num = it._feature_per_frame.size();
        if (it._used_num >= 2 && it._start_frame < svar.GetInt("window_size", 20) - 2)
            cnt++;
    }
    return cnt;
}

void FeatureManager::ClearDepth(const Eigen::VectorXd &x)
{
    int feature_index = -1;
    for(auto &it_per_id: _feature){
        it_per_id._used_num = it_per_id._feature_per_frame.size();
        if( !(it_per_id._used_num >= 2 && it_per_id._start_frame < svar.GetInt("window_size", 20) - 2))
            continue;
        it_per_id._estimated_depth = 1.0 / x(++feature_index);
    }
}

void FeatureManager::Triangulate(std::vector<Eigen::Vector3d> &Ps, std::vector<Eigen::Vector3d> &tic,
                                 std::vector<Eigen::Matrix3d> &ric)
{
    for(auto& it_per_id: _feature){
        // 每个id的特征点被多少帧图像观测到了.
        it_per_id._used_num = it_per_id._feature_per_frame.size();
        // 如果该特征点被两帧及两帧以上的图像观测到
        // 且观测到该特征点的第一帧图像应该早于或等于滑动窗口第4最新关键帧.
        // 也就是说至少是第4最新关键帧和第3最新关键帧观测到了该特征点(第2最新帧似乎是紧耦合优化的最新帧)
        if(!(it_per_id._used_num >= 2 && it_per_id._start_frame < svar.GetInt("window_size", 20) - 2))
            continue;
        // 该id的特征点深度值大于0, 该值初始化的时候为-1, 如果大于0, 则说明该点被三角化过.
        // FIXME: 这里应该可以继续三角化减少误差..
        if(it_per_id._estimated_depth > 0)
            continue;

        // imu_i: 观测到该特征点的第一帧图像在滑动窗口中的帧号.
        // imu_j: 观测到该特征点的最后一帧图像在滑动窗口中的帧号.
        int imu_i = it_per_id._start_frame, imu_j = imu_i - 1;
        Eigen::MatrixXd svd_A(2 * it_per_id._feature_per_frame.size(), 4);
        int svd_idx = 0;
        Eigen::Matrix<double, 3, 4> P0;
        // FIXME: 这里的t0和R0似乎是Imu坐标系, 但是找不到_R被赋值的操作.
        // 如果真的没有初始化,那么_R就是单位阵
        // 由于后面三角化所需的是两两帧之间的位姿变化, 所以这个_R是否赋值没有影响
        Eigen::Vector3d t0 = Ps[imu_i] + _R[imu_i] * tic[0];
        Eigen::Matrix3d R0 = _R[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        // 观测到该id特征点的每一图像帧
        for(auto& it_per_frame: it_per_id._feature_per_frame){
            imu_j++;// 观测到该特征点的最后一帧图像在滑动窗口中的帧号
            Eigen::Vector3d t1 = Ps[imu_j] + _R[imu_j] * tic[0];
            Eigen::Matrix3d R1 = _R[imu_j] * ric[0];
            // t和R为两两帧之间的位姿变化
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame._point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);
        }
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        it_per_id._estimated_depth = svd_method;
        if (it_per_id._estimated_depth < 0.1)
            it_per_id._estimated_depth = para._init_depth;
    }
}

void FeatureManager::RemoveBackShiftDepth(Eigen::Matrix3d &marg_R, Eigen::Vector3d &marg_P, Eigen::Matrix3d &new_R,
                                          Eigen::Vector3d &new_P)
{
    for(auto it = _feature.begin(), it_next = _feature.begin(); it != _feature.end(); it = it_next){
        it_next++;
        if( it->_start_frame != 0 )
            it->_start_frame--;
        else{
            Eigen::Vector3d uv_i = it->_feature_per_frame[0]._point;
            it->_feature_per_frame.erase(it->_feature_per_frame.begin());
            if (it->_feature_per_frame.size() < 2){
                _feature.erase(it);
                continue;
            }
            else{
                Eigen::Vector3d pts_i = uv_i * it->_estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->_estimated_depth = dep_j;
                else
                    it->_estimated_depth = para._init_depth;
            }
        }
    }
}

void FeatureManager::RemoveBack()
{
    for(auto it = _feature.begin(), it_next = _feature.begin(); it != _feature.end(); it = it_next){
        it_next++;
        if(it->_start_frame != 0)
            it->_start_frame--;
        else{
            it->_feature_per_frame.erase(it->_feature_per_frame.begin());
            if(it->_feature_per_frame.size() == 0)
                _feature.erase(it);
        }
    }
}

void FeatureManager::RemoveFront(int frame_count)
{
    for(auto it = _feature.begin(), it_next = _feature.begin(); it != _feature.end(); it = it_next){
        it_next++;
        if (it->_start_frame == frame_count)
            it->_start_frame--;
        else{
            int j = svar.GetInt("window_size", 20) - 1 - it->_start_frame;
            if (it->EndFrame() < frame_count - 1)
                continue;
            it->_feature_per_frame.erase(it->_feature_per_frame.begin() + j);
            if (it->_feature_per_frame.empty())
                _feature.erase(it);
        }
    }
}

void FeatureManager::SetDepth(const Eigen::VectorXd &x)
{
    int feature_index = -1;
    for(auto &it_per_id: _feature){
        it_per_id._used_num = it_per_id._feature_per_frame.size();
        if(!(it_per_id._used_num >= 2 && it_per_id._start_frame < svar.GetInt("window_size", 20) - 2))
            continue;

        it_per_id._estimated_depth = 1.0 / x(++feature_index);
        if(it_per_id._estimated_depth < 0)
            it_per_id._solve_flag = 2;
        else
            it_per_id._solve_flag = 1;
    }
}

void FeatureManager::RemoveFailures()
{
    for(auto it = _feature.begin(), it_next = _feature.begin(); it != _feature.end(); it = it_next){
        it_next++;
        if (it->_solve_flag == 2)
            _feature.erase(it);
    }
}
