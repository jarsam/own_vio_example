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
        FeaturePerFrame feature_per_frame(id_pts.second[0], td);
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
}