//
// Created by liu on 19-12-4.
//

#include <GSLAM/core/GSLAM.h>

#include "FeatureTracker.h"

bool FeatureTracker::ReadImage(const cv::Mat &image, double cur_time)
{
    cv::Mat img;
    _cur_time = cur_time;

    if(svar.GetInt("equalize_image", 1)){
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(image, img);
    }
    else
        img = image;

    if (_forw_img.empty()){
        _prev_img = img;
        _cur_img = img;
        _forw_img = img;
    }
    else{
        _forw_img = img;
    }

    _forw_pts.clear();
    if (_cur_pts.size() > 0){
        std::vector<uchar > status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(_cur_img, _forw_img, _cur_pts, _forw_pts, status, err, cv::Size(21, 21), 3);

        for(int i = 0; i < int(_forw_pts.size()); ++i){
            if (status[i] && !InBoard(_forw_pts[i]))
                status[i] = 0;
        }

//        ReduceVector(_prev_pts, status);
        ReduceVector(_ids, status);
        ReduceVector(_cur_un_pts, status);
        ReduceVector(_track_cnt, status);
        ReduceVector(_cur_pts, status);
        ReduceVector(_forw_pts, status);
    }

    // 如果跟踪到了则跟踪数量加1
    for(auto &n: _track_cnt)
        n++;
    if (para._pub_this_frame){
        RejectWithF();
        //后面要检测新的特征点,所以这里
        SetMask();

        int rest_max_feature = svar.GetInt("max_feature", 150) - _forw_pts.size();
        if (rest_max_feature > 0){
            cv::goodFeaturesToTrack(_forw_img, _features, rest_max_feature, 0.01,
                svar.GetDouble("min_dist", 30.0), _mask);
        }
        else
            _features.clear();

        AddPoints();
    }

    _prev_img = _cur_img;
    _prev_pts = _cur_pts;
    _prev_un_pts = _cur_un_pts;
    _cur_img = _forw_img;
    _cur_pts = _forw_pts;
    //这里是_forw_pts中的特征点进行去畸变. 并且还跟踪了一些新的特征点.
    //FIXME:但是在RejectWithF函数中已经去畸变了,所以感觉没必要,但是可能之后会用到新的特征点,所以要重新去畸变.
    UndistortedPoints();
    ComputePointsVelocity();
    _prev_time = _cur_time;
}

