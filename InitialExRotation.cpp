//
// Created by liu on 19-12-10.
//

#include "InitialExRotation.h"

InitialExRotation::InitialExRotation()
{
    _frame_count = 0;
    _Rc.push_back(Eigen::Matrix3d::Identity());
    _Rc_g.push_back(Eigen::Matrix3d::Identity());
    _Rimu.push_back(Eigen::Matrix3d::Identity());
    _ric = Eigen::Matrix3d::Identity();
}

bool InitialExRotation::CalibrationExRotation(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres,
                                              Eigen::Quaterniond &delta_q_imu, Eigen::Matrix3d &calib_ric_result)
{
    _frame_count++;
    _Rc.emplace_back(SolveRelativeR(corres));
    _Rimu.emplace_back(delta_q_imu.toRotationMatrix());
    _Rc_g.emplace_back(_ric.inverse() * delta_q_imu * _ric);

    Eigen::MatrixXd A(_frame_count * 4, 4);
}

Eigen::Matrix3d InitialExRotation::SolveRelativeR(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres)
{
    if(corres.size() >= 9){
        std::vector<cv::Point2f> ll, rr;
        for (int i = 0; i < corres.size(); ++i){
            ll.emplace_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.emplace_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }

        cv::Mat E = cv::findFundamentalMat(ll, rr);
        cv::Mat_<double> R1, R2, t1, t2;
        DecomposeE(E, R1, R2, t1, t2);

        if (cv::determinant(R1) + 1.0 < 1e-09){
            E = -E;
            DecomposeE(E, R1, R2, t1, t2);
        }

        double ratio1 = std::max(TestTriangulation(ll, rr, R1, t1), TestTriangulation(ll, rr, R1, t2));
        double ratio2 = std::max(TestTriangulation(ll, rr, R2, t1), TestTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;

        Eigen::Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j);
        return ans_R_eigen;
    }
}

void InitialExRotation::DecomposeE(cv::Mat E, cv::Mat_<double> &R1, cv::Mat_<double> &R2, cv::Mat_<double> &t1,
                                   cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}

double InitialExRotation::TestTriangulation(const std::vector<cv::Point2f> &l, const std::vector<cv::Point2f> &r,
                                            cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat point_cloud;
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    cv::triangulatePoints(P, P1, l, r, point_cloud);
    int front_count = 0;
    for (int i = 0; i < point_cloud.cols; i++)
    {
        double normal_factor = point_cloud.col(i).at<float>(3);

        cv::Mat_<double> p_3d_l = cv::Mat(P) * (point_cloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (point_cloud.col(i) / normal_factor);
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    return 1.0 * front_count / point_cloud.cols;
}
