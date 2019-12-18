//
// Created by liu on 19-12-18.
//

#ifndef VIO_EXAMPLE_POSELOCALPARAMETERIZATION_H
#define VIO_EXAMPLE_POSELOCALPARAMETERIZATION_H

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <Utility.h>

class PoseLocalParameterization: public ceres::LocalParameterization
{
    // 传入的x为[p, q], 一共有7位, 而delta只有6位.
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const{
        Eigen::Map<const Eigen::Vector3d> _p(x);
        Eigen::Map<const Eigen::Quaterniond> _q(x + 3);
        Eigen::Map<const Eigen::Vector3d> dp(delta);
        Eigen::Quaterniond dq = Utility::DeltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));
        Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
        Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);
        p = _p + dp;
        q = (_q * dq).normalized();
        return true;
    }
    virtual bool ComputeJacobian(const double *x, double *jacobian) const{

    }
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };
};


#endif //VIO_EXAMPLE_POSELOCALPARAMETERIZATION_H
