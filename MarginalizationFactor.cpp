//
// Created by liu on 19-12-18.
//

#include "MarginalizationFactor.h"

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

MarginalizationInfo::~MarginalizationInfo()
{
    for(auto it = _parameter_block_data.begin(); it != _parameter_block_data.end(); ++it)
        delete[] it->second;

    for(int i = 0; i < _factors.size(); ++i){
        delete[] _factors[i]->_raw_jacobians;
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

// 调用每个Factor的Evaluate函数, 对_parameter_block_data进行赋值
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

void* ThreadConstructA(void* thread_struct)
{
    ThreadStruct* p = ((ThreadStruct*)thread_struct);
    for(auto it: p->_sub_factors){
        for(int i = 0; i < it->_parameter_blocks.size(); ++i){
            int idx_i = p->_parameter_block_idx[reinterpret_cast<long>(it->_parameter_blocks[i])];
            int size_i = p->_parameter_block_size[reinterpret_cast<long>(it->_parameter_blocks[i])];
            if (size_i == 7)
                size_i = 6;
            Eigen::MatrixXd jacobian_i = it->_jacobians[i].leftCols(size_i);
            for(int j = i; j < it->_parameter_blocks.size(); ++j){
                int idx_j = p->_parameter_block_idx[reinterpret_cast<long>(it->_parameter_blocks[j])];
                int size_j = p->_parameter_block_size[reinterpret_cast<long>(it->_parameter_blocks[j])];
                if(size_j == 7)
                    size_j = 6;
                Eigen::MatrixXd jacobian_j = it->_jacobians[j].leftCols(size_j);
                if(i == j)
                    p->_A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else{
                    p->_A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->_A.block(idx_j, idx_i, size_j, size_i) = p->_A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->_b.segment(idx_i, size_i) += jacobian_i.transpose() * it->_residuals;
        }
    }

    return thread_struct;
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

    std::vector<pthread_t> threads_id(svar.GetInt("thread_number", 4));
    std::vector<ThreadStruct> threads_struct(svar.GetInt("thread_number", 4));
    int i = 0;
    for(auto it: _factors){
        threads_struct[i]._sub_factors.emplace_back(it);
        ++i;
        i = i % svar.GetInt("thread_number", 4);
    }
    for(int i = 0; i < svar.GetInt("thread_number", 4); ++i){
        threads_struct[i]._A = Eigen::MatrixXd::Zero(pos, pos);
        threads_struct[i]._b = Eigen::VectorXd::Zero(pos);
        threads_struct[i]._parameter_block_size = _parameter_block_size;
        threads_struct[i]._parameter_block_idx = _parameter_block_idx;
        int ret = pthread_create(&threads_id[i], NULL, ThreadConstructA, (void*)&(threads_struct[i]));
    }
    for(int i = svar.GetInt("thread_number", 4) - 1; i >= 0; --i){
        pthread_join(threads_id[i], NULL);
        A += threads_struct[i]._A;
        b += threads_struct[i]._b;
    }

    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, _m, _m) + A.block(0, 0, _m, _m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() *
        Eigen::VectorXd((saes.eigenvalues().array() > _eps).select(saes.eigenvalues().array().inverse(), 0))).asDiagonal() *
        saes.eigenvectors().transpose();

    Eigen::VectorXd bmm = b.segment(0, _m);
    Eigen::MatrixXd Amr = A.block(0, _m, _m, _n);
    Eigen::MatrixXd Arm = A.block(_m, 0, _n, _m);
    Eigen::MatrixXd Arr = A.block(_m, _m, _n, _n);
    Eigen::VectorXd brr = b.segment(_m, _n);
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > _eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > _eps).select(saes2.eigenvalues().array().inverse(), 0));
    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
    // FIXME: 下面这两个是干嘛的?
    _linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvalues().transpose();
    _linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
}

std::vector<double *> MarginalizationInfo::GetParameterBlocks(
    std::unordered_map<long, std::vector<double> > &addr_shift)
{
    std::vector<double *> keep_block_addr;
    _keep_block_size.clear();
    _keep_block_idx.clear();
    _keep_block_data.clear();

    for(const auto &it: _parameter_block_idx){
        if(it.second >= _m){
            _keep_block_size.emplace_back(_parameter_block_size[it.first]);
            _keep_block_idx.emplace_back(_parameter_block_idx[it.first]);
            _keep_block_data.emplace_back(_parameter_block_data[it.first]);
            keep_block_addr.emplace_back(addr_shift[it.first]);
        }
    }

    _sum_block_size = std::accumulate(std::begin(_keep_block_size), std::end(_keep_block_size), 0);
    return keep_block_addr;
}
