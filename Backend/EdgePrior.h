//
// Created by liu on 20-1-7.
//

#ifndef VIO_EXAMPLE_EDGEPRIOR_H
#define VIO_EXAMPLE_EDGEPRIOR_H

#include <memory>
#include <string>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include "EigenTypes.h"
#include "Edge.h"

class EdgeSE3Prior: public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE3Prior(const Vec3 &p, const Qd &q): Edge(6, 1, std::vector<std::string>{"VertexPose"}), _Pp(p), _Qp(q){}

    virtual std::string TypeInfo() const override {return "EdgeSE3Prior";}

    virtual void ComputeResidual() override{
        VecX param_i = _verticies[0]->Parameters();
        Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
        Vec3 Pi = param_i.head<3>();

        // 使用了SO3作数据差值, 再转到李代数上
        Sophus::SO3d ri(Qi);
        Sophus::SO3d rp(_Qp);
        Sophus::SO3d res_r = rp.inverse() * ri;
        _residual.block<3, 1>(0, 0) = Sophus::SO3d::log(res_r);
        _residual.block<3, 1>(3, 0) = Pi - _Pp;
    }

    virtual void ComputeJacobians() override{
        VecX param_i = _verticies[0]->Parameters();
        Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);

        Eigen::Matrix<double, 6, 6> jacobian_pose_i = Eigen::Matrix<double, 6, 6>::Zero();

        Sophus::SO3d ri(Qi);
        Sophus::SO3d rp(_Qp);
        Sophus::SO3d res_r = rp.inverse() * ri;

        jacobian_pose_i.block<3, 3>(0, 3) = Sophus::SO3d::JacobianRInv(res_r.log());
        jacobian_pose_i.block<3, 3>(3, 0) = Mat33::Identity();

        _jacobians[0] = jacobian_pose_i;
    }

private:
    Vec3 _Pp; // pose prior
    Qd _Qp; // rotation prior
};

#endif //VIO_EXAMPLE_EDGEPRIOR_H
