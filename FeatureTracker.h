//
// Created by liu on 19-12-4.
//

#ifndef VIO_EXAMPLE_FEATURETRACKER_H
#define VIO_EXAMPLE_FEATURETRACKER_H

#include "Parameters.h"
#include "PinholeCamera.h"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

class FeatureTracker
{
public:
    FeatureTracker(){
        _pinhole_camera.ReadIntrinsicParameters();
    }

    bool ReadImage(const cv::Mat &image, double cur_time);

private:
    // 保证pt在图像中除了border的一圈内.
    bool InBoard(const cv::Point2d &pt){
        const int border_size = 1;
        int img_x = cvRound(pt.x);
        int img_y = cvRound(pt.y);
        return border_size <= img_x && img_x < para._width - border_size && border_size <= img_y  && img_y < para._height - border_size;
    }

    void ReduceVector(std::vector<cv::Point2d> &v, std::vector<uchar> &status){
        int j = 0;
        for(int i = 0; i < int(v.size()); ++i){
            if (status[i])
                v[j++] = v[i];
        }
        v.resize(j);
    }

    void RejectWithF(){
        if (_forw_pts.size() >= 8){
            std::vector<cv::Point2d> un_cur_pts(_cur_pts.size()), un_forw_pts(_forw_pts.size());
            for(int i = 0; i < _cur_pts.size(); ++i){
                Eigen::Vector3d tem_p;
            }
        }
    }
private:
    cv::Mat _prev_img, _cur_img, _forw_img;
    std::vector<cv::Point2d> _prev_pts, _cur_pts, _forw_pts;
    std::vector<cv::Point2d>  _cur_un_pts;
    std::vector<int> _ids, _track_cnt;
    PinholeCamera _pinhole_camera;

    double _cur_time;
};


#endif //VIO_EXAMPLE_FEATURETRACKER_H
