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