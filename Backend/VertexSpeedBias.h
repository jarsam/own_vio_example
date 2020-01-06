//
// Created by liu on 20-1-6.
//

#ifndef VIO_EXAMPLE_VERTEXSPEEDBIAS_H
#define VIO_EXAMPLE_VERTEXSPEEDBIAS_H

#include "Vertex.h"

class VertexSpeedBias: public Vertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexSpeedBias(): Vertex(9) {}
    std::string TypeInfo() const
    {
        return "VertexSpeedBias";
    }
};

#endif //VIO_EXAMPLE_VERTEXSPEEDBIAS_H
