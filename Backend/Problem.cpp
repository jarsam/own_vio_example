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
    if (_verticies.find(vertex->Id()) != _verticies.end()){
        return false;
    }
    else{
        _verticies.insert(std::pair<unsigned long, std::shared_ptr<Vertex> >(vertex->Id(), vertex));
    }

    if (_problem_type == ProblemType::SLAM_PROBLEM){
        if (IsPoseVertex(vertex)){
            ResizePoseHessiansWhenAddingPose(vertex);
        }
    }
}

// remove顶点的同时也会remove包含顶点的边
bool Problem::RemoveVertex(std::shared_ptr<Vertex> vertex)
{
    if (_verticies.find(vertex->Id()) == _verticies.end())
        return false;

    // 这里要remove该顶点对应的edge
    std::vector<std::shared_ptr<Edge> > remove_edges = GetConnectedEdges(vertex);
    for(int i = 0; i < remove_edges.size(); ++i)
        RemoveEdge(remove_edges[i]);

    if(IsPoseVertex(vertex))
        _idx_pose_verticies.erase(vertex->Id());
    else
        _idx_landmark_verticies.erase(vertex->Id());

    vertex->SetOrderingId(-1);
    _verticies.erase(vertex->Id());
    _vertexToEdge.erase(vertex->Id());
}

bool Problem::AddEdge(std::shared_ptr<Edge> edge)
{
    if (_edges.find(edge->Id()) == _edges.end())
        _edges.insert(std::pair<unsigned long, std::shared_ptr<Edge> >(edge->Id(), edge));
    else
        return false;

    for(auto &vertex: edge->Verticies())
        _vertexToEdge.insert(std::pair<unsigned long, std::shared_ptr<Edge> >(vertex->Id(), edge));

    return true;
}

bool Problem::RemoveEdge(std::shared_ptr<Edge> edge)
{
    if (_edges.find(edge->Id()) == _edges.end())
        return false;

    _edges.erase(edge->Id());
    return true;
}

void Problem::ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v)
{
    int size = _H_prior.rows() + v->LocalDimension();
    _H_prior.conservativeResize(size, size);
    _b_prior.conservativeResize(size);

    _b_prior.tail(v->LocalDimension()).setZero();
    _H_prior.rightCols(v->LocalDimension()).setZero();
    _H_prior.bottomRows(v->LocalDimension()).setZero();
}

std::vector<std::shared_ptr<Edge> > Problem::GetConnectedEdges(std::shared_ptr<Vertex> vertex)
{
    std::vector<std::shared_ptr<Edge> > edges;
    auto range = _vertexToEdge.equal_range(vertex->Id());

    for(auto iter = range.first; iter != range.second; ++iter){
        // 这条边没被remove
        if (_edges.find(iter->second->Id()) == _edges.end())
            continue;
        edges.emplace_back(iter->second);
    }

    return edges;
}
