//
// Created by liu on 19-12-4.
//

#include "PinholeCamera.h"

// 像素坐标系转归一化坐标系
void PinholeCamera::LiftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P)
{
    double pre_distort_x = _inv_K11 * p(0) + _inv_K13;
    double pre_distort_y = _inv_K22 * p(1) + _inv_K23;

    int n = 8;
    Eigen::Vector2d distort_pt;
    double distorted_u = 0, distorted_v = 0;
    Distortion(Eigen::Vector2d(pre_distort_x, pre_distort_y), distort_pt);
    distorted_u = pre_distort_x - distort_pt(0);
    distorted_v = pre_distort_y - distort_pt(1);

    //FIXME: 看不懂这个是什么用,感觉没什么用.
    for (int i = 1; i < n; ++i){
        Distortion(Eigen::Vector2d(distorted_u, distorted_v), distort_pt);
        distorted_u = pre_distort_x - distort_pt(0);
        distorted_v = pre_distort_x - distort_pt(1);
    }

    P << distorted_u, distorted_v, 1.0;
}


// 这里值算了畸变的变化量,没有算出畸变后的点坐标.
void PinholeCamera::Distortion(const Eigen::Vector2d &pre_pt, Eigen::Vector2d &norm_pt)
{
    double k1 = _distortion_coefficients[0];
    double k2 = _distortion_coefficients[1];
    double p1 = _distortion_coefficients[2];
    double p2 = _distortion_coefficients[3];

    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = pre_pt(0) * pre_pt(0);
    my2_u = pre_pt(1) * pre_pt(1);
    mxy_u = pre_pt(0) * pre_pt(1);
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    norm_pt << pre_pt(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
        pre_pt(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}