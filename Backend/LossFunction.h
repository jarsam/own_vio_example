//
// Created by liu on 20-1-6.
//

#ifndef VIO_EXAMPLE_LOSSFUNCTION_H
#define VIO_EXAMPLE_LOSSFUNCTION_H

#include "EigenTypes.h"

/*
 * 输出的rho的意思是:
 * rho[0]: 误差的平方
 * rho[1]: 将误差代入到误差函数中的一阶倒数 (注意这里是对误差的平方求导)
 * rho[2]: 将误差代入到误差函数中的二阶倒数
 */
class LossFunction
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    virtual ~LossFunction(){}
    virtual void Compute(double err2, Eigen::Vector3d& rho) const = 0;
};

/*
 * 平凡的loss, 不做任何处理
 * 和使用nullptr作为lossfunction时效果相同
 *
 * TrivalLoss(e) = e^2
 */
class TrivalLoss: public LossFunction
{
public:
    virtual void Compute(double err2, Eigen::Vector3d& rho) const override
    {
        rho[0] = err2;
        rho[1] = 1;
        rho[2] = 0;
    }
};

/*
 * Huber(e) = e^2                  if e <= delta
 * Huber(e) = delta*(2*e - delta)  if e > delta
 *
 * HuberLoss是对于高error值有抑制作用, 但对于小于delta的值则没有任何处理.
 */
class HuberLoss: public LossFunction
{
public:
    explicit HuberLoss(double delta): _delta(delta){}

    virtual void Compute(double err2, Eigen::Vector3d& rho) const override
    {
        double delta_sqr = _delta * _delta;
        if (err2 <= delta_sqr){ // inlier
            rho[0] = err2;
            rho[1] = 1.;
            rho[2] = 0.;
        }
        else{ // outlier
            double sqrte = sqrt(err2); // absolute value of the error
            rho[0] = 2 * sqrte * _delta - delta_sqr;
            rho[1] = 2 * sqrte;
            rho[2] = -0.5 * rho[1] / err2;
        }
    }

private:
    double _delta;
};

/*
 * Cauthy 相较于HuberLoss而言, 对于高误差值项有更强的抑制作用,
 * 并且如果调整delta值, 则能让较小的误差值变得更大, 让较大的误差值变得更小.
 * 但这也有可能带来误差
 */
class CauthyLoss: public LossFunction
{
public:
    explicit CauthyLoss(double delta): _delta(delta){}
    virtual void Compute(double err2, Eigen::Vector3d &rho)const override
    {
        double delta_sqr = _delta * _delta; // c^2
        double inv_delta_sqr = 1. / delta_sqr; // 1/c^2
        double aux = inv_delta_sqr * err2 + 1.0; // 1 + e^2/c^2
        rho[0] = delta_sqr * log(aux); // c^2 * log(1 + e^2/c^2)
        rho[1] = 1. / aux;
        rho[2] = -inv_delta_sqr * std::pow(rho[1], 2);
    }

private:
    double _delta;
};

/*
 * 对误差超过了delta的值直接将误差平方赋值为delta^2/3
 * 对0和delta附近的值有一个抑制, 0-delta中间的值是平滑的
 * 感觉对slam来说没啥用..
 */
class TukeyLoss: public LossFunction
{
public:
    explicit TukeyLoss(double delta): _delta(delta){}
    virtual void Compute(double err2, Eigen::Vector3d &rho) const override
    {
        const double e = sqrt(err2);
        const double delta2 = _delta * _delta;
        if (e <= _delta){
            const double aux = err2 / delta2;
            rho[0] = delta2 * (1. - std::pow((1. - aux), 3)) / 3.;
            rho[1] = std::pow((1. - aux), 2);
            rho[2] = -2. * (1. - aux) / delta2;
        }
        else{
            rho[0] = delta2 / 3;
            rho[1] = 0;
            rho[2] = 0;
        }
    }

private:
    double _delta;
};

#endif //VIO_EXAMPLE_LOSSFUNCTION_H
