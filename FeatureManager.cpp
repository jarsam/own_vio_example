//
// Created by liu on 19-12-6.
//

#include "FeatureManager.h"

FeatureManager::FeatureManager(std::vector<Eigen::Matrix3d> &Rs): _R(Rs)
{
    for(int i = 0; i < svar.GetInt("number_of_camera", 1); ++i)
        _ric.emplace_back(Eigen::Matrix3d::Identity());
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
        return parallax_sum / parallax_num >= svar.GetDouble("min_parallax", 10);
}

// 这个函数实际上是求取该特征点在两帧的归一化平面上的坐标点的距离
double FeatureManager::CompensatedParallax(const FeaturePerId &it_per_id, int frame_count)
{
    const FeaturePerFrame &frame_i = it_per_id._feature_per_frame[frame_count - 2 - it_per_id._start_frame];
    const FeaturePerFrame &frame_j = it_per_id._feature_per_frame[frame_count - 1 - it_per_id._start_frame];

    Eigen::Vector3d p_j = frame_j._point;
    Eigen::Vector3d p_i = frame_i._point;

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