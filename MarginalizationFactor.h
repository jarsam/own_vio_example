//
// Created by liu on 19-12-18.
//

#ifndef VIO_EXAMPLE_MARGINALIZAITONFACTOR_H
#define VIO_EXAMPLE_MARGINALIZAITONFACTOR_H

#include <cstdlib>
#include <pthread.h>
#include <unordered_map>

#include <ceres/ceres.h>
#include <GSLAM/core/GSLAM.h>

#include "Utility.h"

// 将所有的Imu误差函数, 视觉误差函数都包装成这个类
class ResidualBlockInfo
{
public:
    ResidualBlockInfo(std::shared_ptr<ceres::CostFunction> cost_function, std::shared_ptr<ceres::LossFunction> loss_function,
                      std::vector<double *> parameter_block, std::vector<int> drop_set) :
                      _cost_function(cost_function), _loss_function(loss_function),
                      _parameter_blocks(parameter_block), _drop_set(drop_set){}

    void Evaluate();

    std::shared_ptr<ceres::CostFunction> _cost_function;
    std::shared_ptr<ceres::LossFunction> _loss_function;
    std::vector<double*> _parameter_blocks;// 传入的参数
    std::vector<int> _drop_set;// 待marg的优化变量id

    double **_raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > _jacobians;
    Eigen::VectorXd _residuals;

    int LocalSize(int size){
        return size == 7 ? 6 : size;
    }
};

struct ThreadStruct
{
    std::vector<std::shared_ptr<ResidualBlockInfo> > _sub_factors;
    Eigen::MatrixXd _A;
    Eigen::VectorXd _b;
    std::unordered_map<long, int> _parameter_block_size;
    std::unordered_map<long, int> _parameter_block_idx;
};

// 存入所有的误差函数
class MarginalizationInfo
{
public:
    ~MarginalizationInfo();
    int LocalSize(int size) const{
        return size == 7 ? 6 : size;
    }
    void AddResidualBlockInfo(std::shared_ptr<ResidualBlockInfo> residual_block_info);
    void PreMarginalize();
    void Marginalize();
    std::vector<double *> GetParameterBlocks(std::unordered_map<long, std::vector<double> > &addr_shift);

    std::vector<std::shared_ptr<ResidualBlockInfo> > _factors;
    // _m为要margin掉的变量个数,也就是parameter_block_idx的总localSize, 以double为单位, VBias为9, PQ为6
    // _n为要保留下的优化变量的变量个数, n=localSize(parameter_block_size) - m
    int _m, _n;
    // long代表的是地址, int代表的是size
    std::unordered_map<long, int> _parameter_block_size; // global size
    // long代表地址, int代表参数的位置, 要marg的变量会放在前面
    std::unordered_map<long, int> _parameter_block_idx; // local size
    // long代表的是地址, double *代表的是参数
    std::unordered_map<long, double *> _parameter_block_data;// 每个Factor中的参数

    const double _eps = 1e-8;

    // 先验值的雅克比
    Eigen::MatrixXd _linearized_jacobians;
    Eigen::VectorXd _linearized_residuals;

    std::vector<int> _keep_block_size;// 存的是没marg的变量的大小
    std::vector<int> _keep_block_idx;// 存的是没marg的变量的位置
    std::vector<double *> _keep_block_data; // 存的是没marg的变量的数据

    int _sum_block_size;
};

class MarginalizationFactor: public ceres::CostFunction
{
public:
    MarginalizationFactor(std::shared_ptr<MarginalizationInfo> marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians)const;

    // 这个_marginalization_info是上一次的边缘化信息
    std::shared_ptr<MarginalizationInfo> _marginalization_info;
};


#endif //VIO_EXAMPLE_MARGINALIZAITONFACTOR_H
