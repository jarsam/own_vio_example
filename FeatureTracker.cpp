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
        std::vector<double> err;
        cv::calcOpticalFlowPyrLK(_cur_img, _forw_img, _cur_pts, _forw_pts, status, err, cv::Size(21, 21), 3);

        for(int i = 0; i < int(_forw_pts.size()); ++i){
            if (status[i] && !InBoard(_forw_pts[i]))
                status[i] = 0;
        }

//        ReduceVector(_prev_pts, status);
//        ReduceVector(_ids, status);
//        ReduceVector(_cur_un_pts, status);
//        ReduceVector(_track_cnt, status);
        ReduceVector(_cur_pts, status);
        ReduceVector(_forw_pts, status);
    }

    for(auto &n: _track_cnt)
        n++;
    if (para._pub_this_frame){
        RejectWithF();
    }
}

