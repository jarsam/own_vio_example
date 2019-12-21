//
// Created by liu on 19-12-18.
//

#include "MarginalizaitonFactor.h"

// 这个函数就将参数传入原本的Evaluate函数中
void ResidualBlockInfo::Evaluate()
{
    _residuals.resize(_cost_function->num_residuals());
    std::vector<int> block_sizes = _cost_function->parameter_block_sizes();
    _raw_jacobians = new double *[block_sizes.size()];
    _jacobians.resize(block_sizes.size());

    for(int i = 0; i < block_sizes.size(); ++i){
        _jacobians[i].resize(_cost_function->num_residuals(), block_sizes[i]);
        _raw_jacobians[i] = _jacobians[i].data();
    }
    _cost_function->Evaluate(_parameter_blocks.data(), _residuals.data(), _raw_jacobians);

    if (_loss_function){

    }
}

void MarginalizationInfo::AddResidualBlockInfo(std::shared_ptr<ResidualBlockInfo> residual_block_info)
{
    _factors.emplace_back(residual_block_info);
    std::vector<double *> &parameter_blocks = residual_block_info->_parameter_blocks;
    std::vector<int> parameter_block_sizes = residual_block_info->_cost_function->parameter_block_sizes();

    for(int i = 0; i < residual_block_info->_parameter_blocks.size(); ++i){
        double *addr = parameter_blocks[i];
        int size = parameter_block_sizes[i];
        _parameter_block_size[reinterpret_cast<long>(addr)] = size;
    }
    for(int i = 0; i < residual_block_info->_drop_set.size(); ++i){
        double *addr = parameter_blocks[residual_block_info->_drop_set[i]];
        _parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}

// 调用每个Factor的Evaluate函数, 进行
void MarginalizationInfo::PreMarginalize()
{
    for(auto it: _factors){
        it->Evaluate();

        std::vector<int> block_sizes = it->_cost_function->parameter_block_sizes();
        for(int i = 0; i < block_sizes.size(); ++i){
            long addr = reinterpret_cast<long>(it->_parameter_blocks[i]);
            int size = block_sizes[i];
            if(_parameter_block_data.find(addr) == _parameter_block_data.end()){
                double *data = new double[size];
                memcpy(data, it->_parameter_blocks[i], sizeof(double) * size);
                _parameter_block_data[addr] = data;
            }
        }
    }
}

void MarginalizationInfo::Marginalize()
{
    int pos = 0;
    for(auto &it: _parameter_block_idx){
        it.second = pos;
        pos += LocalSize(_parameter_block_size[it.first]);
    }
    _m = pos;

    for(const auto &it: _parameter_block_size){
        if(_parameter_block_idx.find(it.first) == _parameter_block_idx.end()){
            _parameter_block_idx[it.first] = pos;
            pos += LocalSize(it.second);
        }
    }

    _n = pos - _m;

    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();
}
