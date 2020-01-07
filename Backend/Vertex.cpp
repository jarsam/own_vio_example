//
// Created by liu on 20-1-6.
//

#include "Vertex.h"

unsigned long global_vertex_id = 0;

Vertex::Vertex(int num_dimension, int local_dimension)
{
    _parameters.resize(num_dimension, 1);
    _local_dimension = local_dimension > 0 ? local_dimension : num_dimension;
    _id = global_vertex_id++;
}

Vertex::~Vertex() {}

int Vertex::Dimension() const
{
    return _parameters.rows();
}

int Vertex::LocalDimension() const
{
    return _local_dimension;
}

void Vertex::Plus(const VecX &delta)
{
    _parameters += delta;
}
