//
// Created by liu on 20-1-7.
//

#ifndef VIO_EXAMPLE_VERTEXINVERSEDEPTH_H
#define VIO_EXAMPLE_VERTEXINVERSEDEPTH_H

#include "Vertex.h"

class VertexInverseDepth: public Vertex{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexInverseDepth(): Vertex(1) {}
    virtual std::string TypeInfo() const {return "VertexInverseDepth";}
};

#endif //VIO_EXAMPLE_VERTEXINVERSEDEPTH_H
