//
// Created by liu on 19-12-4.
//

#ifndef VIO_EXAMPLE_FEATURETRACKER_H
#define VIO_EXAMPLE_FEATURETRACKER_H

#include "Parameters.h"
#include "PinholeCamera.h"

#include <iostream>
#include <map>
#include <list>

#include <Eigen/Dense>
#include <GSLAM/core/GSLAM.h>
#include <opencv2/opencv.hpp>

class FeatureTracker
{
public:
    FeatureTracker(){
        _pinhole_camera.ReadIntrinsicParameters();
    }

    bool ReadImage(const cv::Mat &image, double cur_time);
    bool UpdataID(int i){
        if (i < _ids.size()){
            if (_ids[i] == -1)
                _ids[i] = _id++;
            return true;
        }
        else
            return false;
    }
    bool AddFeatureCheckParallax(int frame_count, const std::map<int, std::vector<int, Eigen::Matrix<double, 7, 1>>> &image,
                                 double td);

private:
    // 保证pt在图像中除了border的一圈内.
    bool InBoard(const cv::Point2d &pt){
        const int border_size = 1;
        int img_x = cvRound(pt.x);
        int img_y = cvRound(pt.y);
        return border_size <= img_x && img_x < para._width - border_size && border_size <= img_y  && img_y < para._height - border_size;
    }

    template <class T>
    void ReduceVector(std::vector<T> &v, std::vector<uchar> &status){
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
                Eigen::Vector3d tem_pt;
                // 首先去畸变,然后用fundamental矩阵进行RANSAC.
                //
                _pinhole_camera.LiftProjective(Eigen::Vector2d(_cur_pts[i].x, _cur_pts[i].y), tem_pt);
                tem_pt.x() = para._camera_intrinsics[0] * tem_pt.x() + para._camera_intrinsics[2];
                tem_pt.y() = para._camera_intrinsics[1] * tem_pt.y() + para._camera_intrinsics[3];
                un_cur_pts[i] = cv::Point2d(tem_pt.x(), tem_pt.y());

//                LOG(INFO) << "pre_pt: " << _cur_pts[i];
//                LOG(INFO) << "tem_pt: " << tem_pt;
                _pinhole_camera.LiftProjective(Eigen::Vector2d(_forw_pts[i].x, _forw_pts[i].y), tem_pt);
                tem_pt.x() = para._camera_intrinsics[0] * tem_pt.x() + para._camera_intrinsics[2];
                tem_pt.y() = para._camera_intrinsics[1] * tem_pt.y() + para._camera_intrinsics[3];
                un_forw_pts[i] = cv::Point2d(tem_pt.x(), tem_pt.y());
            }

            std::vector<uchar> status;
            cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC,
                svar.GetDouble("fundamental_threshold", 1.0), 0.99, status);
            int cur_pt_size = _cur_pts.size();
            ReduceVector(_cur_pts, status);
            ReduceVector(_forw_pts, status);
            ReduceVector(_cur_un_pts, status);
//            ReduceVector(_prev_pts, status);
            // 这两行要去除不满足fundamental矩阵的特征点,而_ids和_track_cnt和_cur_pts,_forw_pts,_cur_un_pts是一一对应的.
            ReduceVector(_ids, status);
            ReduceVector(_track_cnt, status);
        }
    }

    void SetMask(){
        _mask = cv::Mat(para._height, para._width, CV_8UC1, cv::Scalar(255));
        std::vector<std::pair<int, std::pair<cv::Point2d, int > > > cnt_pts_id;
        for(int i = 0; i < _forw_pts.size(); ++i)
            cnt_pts_id.emplace_back(std::make_pair(_track_cnt[i], std::make_pair(_forw_pts[i], _ids[i])));
        sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const std::pair<int, std::pair<cv::Point2f, int>> &a, const std::pair<int, std::pair<cv::Point2f, int>> &b)
        {
            return a.first > b.first;
        });
        // 这里按照_track_cnt进行排序,应该是按照跟踪到的个数进行排序的,所以就要将下面的东西都给清理了.
        _forw_pts.clear();
        _ids.clear();
        _track_cnt.clear();

        for(auto &it: cnt_pts_id){
            if(_mask.at<uchar>(it.second.first) == 255){
                _forw_pts.emplace_back(it.second.first);
                _ids.emplace_back(it.second.second);
                _track_cnt.emplace_back(it.first);
                cv::circle(_mask, it.second.first, svar.GetDouble("feature_min_dist", 30.0), 0, -1);
            }
        }
    }

    void AddPoints(){
        // 这里为什么要给_forw_pts赋值就是因为后面会将这些点给_cur_pts
        for(auto &p: _features){
            _forw_pts.emplace_back(p);
            _ids.emplace_back(-1);
            _track_cnt.emplace_back(1);
        }
    }

    void UndistortedPoints(){
        _cur_un_pts.clear();
        _cur_un_pts_map.clear();

        for(int i = 0; i < _cur_pts.size(); ++i){
            Eigen::Vector2d cur_pt(_cur_pts[i].x, _cur_pts[i].y);
            Eigen::Vector3d un_cur_pt;
            _pinhole_camera.LiftProjective(cur_pt, un_cur_pt);
            _cur_un_pts.emplace_back(cv::Point2d(un_cur_pt.x(), un_cur_pt.y()));
            _cur_un_pts_map.insert(std::make_pair(_ids[i], cv::Point2d(un_cur_pt.x(), un_cur_pt.y())));
        }
    }

    //计算点的速度
    void ComputePointsVelocity(){
        if (!_prev_un_pts_map.empty()){
            double dt = _cur_time - _prev_time;
            _pts_velocity.clear();
            for(int i = 0; i < _cur_un_pts.size(); ++i){
                if (_ids[i] != -1){
                    std::map<int, cv::Point2d>::iterator it = _prev_un_pts_map.find(_ids[i]);
                    if (it != _prev_un_pts_map.end()){
                        double v_x = (_cur_un_pts[i].x - it->second.x) / dt;
                        double v_y = (_cur_un_pts[i].y - it->second.y) / dt;
                        _pts_velocity.emplace_back(cv::Point2d(v_x, v_y));
                    }
                    else
                        _pts_velocity.emplace_back(cv::Point2d(0, 0));
                }
                else
                    _pts_velocity.emplace_back(cv::Point2d(0, 0));
            }
        }
        else {
            for(int i = 0; i < _cur_pts.size(); ++i)
                _pts_velocity.emplace_back(cv::Point2d(0, 0));
        }

        _prev_un_pts_map = _cur_un_pts_map;
    }

public:
    cv::Mat _mask;
    cv::Mat _prev_img, _cur_img, _forw_img;
    std::vector<cv::Point2d> _features;

    //在这里_ids和_track_cnt,_cur_pts_,_cur_un_pts都是一一对应的关系.这些点都是未去畸变且是在图像上的点.
    std::vector<cv::Point2f> _prev_pts, _cur_pts, _forw_pts;
    //_cur_un_pts是去畸变的且归一化的特征点.
    std::vector<cv::Point2d> _cur_un_pts, _prev_un_pts;
    std::vector<cv::Point2d> _pts_velocity;
    // _ids首先赋值为-1,然后在system中判断是否为-1赋值为id.
    std::vector<int> _ids, _track_cnt;
    PinholeCamera _pinhole_camera;
    std::map<int, cv::Point2d> _cur_un_pts_map, _prev_un_pts_map;

    double _cur_time, _prev_time;
    int _id = 0;
};


#endif //VIO_EXAMPLE_FEATURETRACKER_H
