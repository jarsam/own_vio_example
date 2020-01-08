//
// Created by liu on 20-1-6.
//

#include "Edge.h"

unsigned long global_edge_id = 0;

Edge::Edge(int residual_dimension, int num_verticies, const std::vector<std::string> &verticies_types)
{
    _residual.resize(residual_dimension, 1);
    if (_verticies_types.empty())
        _verticies_types = verticies_types;

    _jacobians.resize(num_verticies);
    _id = global_edge_id++;

    Eigen::MatrixXd information(residual_dimension, residual_dimension);
    information.setIdentity();
    _information = information;

    _lossfunction = nullptr;
}

double Edge::Chi2() const
{
    return _residual.transpose() * _information * _residual;
}

double Edge::RobustChi2() const
{
    double e2 = this->Chi2();
    if (_lossfunction){
        Eigen::Vector3d rho;
        _lossfunction->Compute(e2, rho);
        e2 = rho[0];
    }
    return e2;
}

// 鲁棒核函数会修改残差和信息矩阵, 如果没有设置robust cost function, 就会返回原来的.
void Edge::RobustInfo(double &drho, MatXX &info) const
{
    if (_lossfunction){
        double e2 = this->Chi2();
        Eigen::Vector3d rho;
        _lossfunction->Compute(e2, rho);
        VecX weight_err = _information * _residual;

        MatXX robust_info(_information.rows(), _information.cols());
        robust_info.setIdentity();
        robust_info *= rho[1] * _information;
        // FIXME: 下面这个公式不是很懂.
        if(rho[1] + 2 * rho[2] * e2 > 0.)
            robust_info += 2 * rho[2] * weight_err * weight_err.transpose();

        info = robust_info;
        drho = rho[1];
    }
    else{
        drho = 1.0;
        info = _information;
    }
}