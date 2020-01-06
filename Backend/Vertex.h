//
// Created by liu on 20-1-6.
//

#ifndef VIO_EXAMPLE_VERTEX_H
#define VIO_EXAMPLE_VERTEX_H

#include "EigenTypes.h"

extern unsigned long global_vertex_id;

class Vertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    explicit Vertex(int num_dimension, int local_dimension = -1);
    virtual ~Vertex();
    int Dimension() const;
    int LocalDimension() const;
    virtual void Plus(const VecX &delta);

    unsigned long Id() const {return _id;}
    /// 返回参数值
    VecX Parameters() const { return _parameters; }
    /// 返回参数值的引用
    VecX &Parameters() { return _parameters; }
    /// 设置参数值
    void SetParameters(const VecX &params) { _parameters = params; }
    // 备份和回滚参数，用于丢弃一些迭代过程中不好的估计
    void BackUpParameters() { _parameters_backup = _parameters; }
    void RollBackParameters() { _parameters = _parameters_backup; }
    virtual std::string TypeInfo() const = 0;
    int OrderingId() const {return _ordering_id;}
    void SetOrderingId(unsigned long id) {_ordering_id = id;}
    void SetFixed(bool fixed = true){_fixed = fixed;}
    bool IsFixed() const {return _fixed;}

protected:
    VecX _parameters; // 实际存储的变量值
    VecX _parameters_backup; // 实际迭代优化中对参数进行备份, 用于回滚.
    int _local_dimension; // 局部参数化维度
    unsigned long _id; // 顶点的id, 自动生成

    // 在problem中排序后的id, 用于寻找雅克比对应块
    // 该值带有维度信息, 例如_ordering_id = 6则对应Hessian中的第6列
    // 从0开始
    unsigned long _ordering_id = 0;
    bool _fixed = false;
};


#endif //VIO_EXAMPLE_VERTEX_H
