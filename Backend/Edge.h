//
// Created by liu on 20-1-6.
//

#ifndef VIO_EXAMPLE_EDGE_H
#define VIO_EXAMPLE_EDGE_H

#include <memory>
#include <string>

#include <Eigen/Dense>

#include "EigenTypes.h"
#include "LossFunction.h"
#include "Vertex.h"
#include "Parameters.h"

/*
 * 边负责计算残差, 残差=预测值-观测值, 维度在构造函数中定义
 * 代价函数=残差*信息矩阵*残差, 是一个数值, 由后端求和最小化
 */
class Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /*
     * 构造函数, 会自动分配雅克比空间
     */
    explicit Edge(int residual_dimension, int num_verticies,
                  const std::vector<std::string> &verticies_types = std::vector<std::string>());
    virtual ~Edge(){};

    unsigned long Id() const {return _id;}

    bool SetVertex(const std::vector<std::shared_ptr<Vertex> > &verticies){
        _verticies = verticies;
        return true;
    }

    std::vector<std::shared_ptr<Vertex> > Verticies() const {
        return _verticies;
    }

    virtual std::string TypeInfo() const = 0;

    // 计算残差
    virtual void ComputeResidual() = 0;

    // 计算雅克比矩阵
    virtual void ComputeJacobians() = 0;

    // 计算平方误差, 会乘以信息矩阵
    double Chi2() const;
    double RobustChi2() const;

    // 返回残差
    VecX Residual() const {return _residual;}

    // 返回雅克比
    std::vector<MatXX> Jacobians() const {return _jacobians;}

    MatXX Information() const {
        return _information;
    }

    MatXX SqrtInformation() const {
        return _sqrt_information;
    }

    void SetInformation(const MatXX &information){
        _information = information;
        _sqrt_information = Eigen::LLT<MatXX>(_information).matrixL().transpose();
    }

    void RobustInfo(double &drho, MatXX &info) const;

    void SetLossFunction(LossFunction* loss){_lossfunction = loss;}

protected:
    unsigned long _id; // edge id
    int _ordering_id; // edge id in problem

    std::vector<std::shared_ptr<Vertex> > _verticies; // 该边对应的顶点
    std::vector<std::string> _verticies_types; // 各顶点类型信息
    VecX _residual; // 残差
    std::vector<MatXX> _jacobians; // 雅克比, 每个雅克比矩阵维度为 residual*vertex[i]
    MatXX _information; // 信息矩阵
    MatXX _sqrt_information;
    VecX _observation; // 观测信息

    LossFunction *_lossfunction;
};


#endif //VIO_EXAMPLE_EDGE_H
