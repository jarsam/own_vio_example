//
// Created by liu on 20-1-6.
//

#ifndef VIO_EXAMPLE_VERTEXPOSE_H
#define VIO_EXAMPLE_VERTEXPOSE_H

#include "Vertex.h"

#include <sophus/se3.hpp>

/*
 * parameters: tx, ty, tz, qx, qy, qz, qw, 7DOF
 * 更新为6自由度的
 * pose表示为Twb
 */
class VertexPose: public Vertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPose(): Vertex(7, 6){}

    virtual  void Plus(const VecX &delta) override
    {
        VecX &parameters = Parameters();
        parameters.head<3>() += delta.head<3>();
        Qd q(parameters[6], parameters[3], parameters[4], parameters[5]);
        q = q * Sophus::SO3d::exp(Vec3(delta[3], delta[4], delta[5])).unit_quaternion();
        q.normalized();
        parameters[3] = q.x();
        parameters[4] = q.y();
        parameters[5] = q.z();
        parameters[6] = q.w();
    }

    std::string TypeInfo() const
    {
        return "VertexPose";
    }
    /**
     * 需要维护[H|b]矩阵中的如下数据块
     * p: pose, m:mappoint
     *
     *     Hp1_p2
     *     Hp2_p2    Hp2_m1    Hp2_m2    Hp2_m3     |    bp2
     *
     *                         Hm2_m2               |    bm2
     *                                   Hm2_m3     |    bm3
     * 1. 若该Camera为source camera，则维护vHessionSourceCamera；
     * 2. 若该Camera为measurement camera, 则维护vHessionMeasurementCamera；
     * 3. 并一直维护m_HessionDiagonal；
     */
};

#endif //VIO_EXAMPLE_VERTEXPOSE_H
