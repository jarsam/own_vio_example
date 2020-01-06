//
// Created by liu on 20-1-6.
//

#include "Problem.h"

Problem::Problem(Problem::ProblemType problem_type): _problem_type(problem_type)
{
    _verticies_marg.clear();
}

Problem::~Problem()
{
    global_vertex_id = 0;
}

bool Problem::AddVertex(std::shared_ptr<Vertex> vertex)
{
    if (_verticies.find(vertex->Id() != _verticies.end())){
        return false;
    }
    else{
        _verticies.insert(std::pair<unsigned long, std::shared_ptr<Vertex> >(vertex->Id(), vertex));
    }

    if (_problem_type == ProblemType::SLAM_PROBLEM){
        if (IsPoseVertex(vertex)){

        }
    }
}
