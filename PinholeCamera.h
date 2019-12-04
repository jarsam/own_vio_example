//
// Created by liu on 19-12-4.
//

#ifndef VIO_EXAMPLE_PINHOLECAMERA_H
#define VIO_EXAMPLE_PINHOLECAMERA_H

#include "Parameters.h"

#include <vector>
#include <Eigen/Dense>

class PinholeCamera
{
public:
    PinholeCamera(){}
    void ReadIntrinsicParameters(){
        _focal_x = para._camera_intrinsics[0];
        _focal_y = para._camera_intrinsics[1];
        _cx = para._camera_intrinsics[2];
        _cy = para._camera_intrinsics[3];
        _width = para._width;
        _height = para._height;
        _distortion_coefficients = para._distortion_coefficients;

        _inv_K11 = 1 /_focal_x;
        _inv_K13 = -_cx / _focal_x;
        _inv_K22 = 1 / _focal_y;
        _inv_K23 = -_cy / _focal_y;
    }

    void LiftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P);

private:
    double _focal_x;
    double _focal_y;
    double _cx;
    double _cy;
    double _width;
    double _height;

    double _inv_K11;
    double _inv_K13;
    double _inv_K22;
    double _inv_K23;
    std::vector<double> _distortion_coefficients;
};


#endif //VIO_EXAMPLE_PINHOLECAMERA_H
