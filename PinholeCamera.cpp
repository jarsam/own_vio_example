//
// Created by liu on 19-12-4.
//

#include "PinholeCamera.h"

// 像素坐标系转归一化坐标系
void PinholeCamera::LiftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P)
{
    double pre_distort_x = _inv_K11 * p(0) + _inv_K13;
    double pre_distort_y = _inv_K22 * p(1) + _inv_K23;


}