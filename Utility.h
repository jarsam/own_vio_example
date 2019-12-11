//
// Created by liu on 19-12-10.
//

#ifndef VIO_EXAMPLE_UTILITY_H
#define VIO_EXAMPLE_UTILITY_H

#include <cmath>
#include <cassert>
#include <cstring>
#include <Eigen/Dense>

class Utility
{
public:
    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> DeltaQ(const Eigen::MatrixBase<Derived> &theta){
        typedef typename  Derived::Scalar Scalar_t;
        Eigen::Quaternion<Scalar_t > dq;
        Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
        half_theta /= static_cast<Scalar_t >(2.0);
        dq.w() = static_cast<Scalar_t >(1.0);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> SkewSymmetric(const Eigen::MatrixBase<Derived> &q){
        Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
        ans << typename Derived::Scalar(0), -q(2), q(1),
            q(2), typename Derived::Scalar(0), -q(0),
            -q(1), q(0), typename Derived::Scalar(0);
        return ans;
    }
};

#endif //VIO_EXAMPLE_UTILITY_H
