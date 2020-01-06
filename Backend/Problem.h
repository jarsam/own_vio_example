//
// Created by liu on 20-1-6.
//

#ifndef VIO_EXAMPLE_PROBLEM_H
#define VIO_EXAMPLE_PROBLEM_H

#include <memory>
#include <map>
#include <unordered_map>

#include "Edge.h"
#include "Vertex.h"
#include "EigenTypes.h"

typedef std::map<unsigned long, std::shared_ptr<Vertex> > HashVertex;
typedef std::unordered_map<unsigned long, std::shared_ptr<Edge> > HashEdge;
typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge> > HashVertexIdToEdge;

class Problem
{
public:
    /*
     * 如果是SLAM问题, 那么pose和landmark是区分开的, Hessian以稀疏方式存储
     * SLAM问题只接受一些特定的Vertex和Edge
     * 如果是通用问题那么Hessian是稠密的, 除非用户设定某些vertex是marginalized
     */
    enum ProblemType{
        SLAM_PROBLEM,
        GENERIC_PROBLEM
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Problem(ProblemType problem_type);
    ~Problem();
    bool AddVertex(std::shared_ptr<Vertex> vertex);

    bool IsPoseVertex(std::shared_ptr<Vertex> v){
        std::string type = v->TypeInfo();
        return type == std::string("VertexPose") || std::string("VertexSpeedBias");
    }

private:
    ProblemType _problem_type;

    // all verticies
    HashVertex _verticies;

    // verticies need to marg
    HashVertex _verticies_marg;
};


#endif //VIO_EXAMPLE_PROBLEM_H
