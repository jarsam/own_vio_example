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

void Problem::ExtendHessiansPriorSize(int dim)
{
    int size = _H_prior.rows() + dim;
    _H_prior.conservativeResize(size, size);
    _b_prior.conservativeResize(size);

    _b_prior.tail(dim).setZero();
    _H_prior.rightCols(dim).setZero();
    _H_prior.bottomRows(dim).setZero();
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

void Problem::AddOrderingSLAM(std::shared_ptr<Vertex> v)
{
    if (IsPoseVertex(v)){
        v->SetOrderingId(_ordering_poses);
        _ordering_poses += v->LocalDimension();
        _idx_pose_verticies.insert(std::pair<unsigned long, std::shared_ptr<Vertex> >(v->Id(), v));
    }
    else if(IsLandmarkVertex(v)){
        v->SetOrderingId(_ordering_landmarks);
        _ordering_landmarks += v->LocalDimension();
        _idx_landmark_verticies.insert(std::make_pair(v->Id(), v));
    }
}

void Problem::SetOrdering()
{
    // 每次都要重新计数
    _ordering_poses = 0;
    _ordering_landmarks = 0;
    _ordering_generic = 0;

    for(auto vertex: _verticies){
        _ordering_generic += vertex.second->LocalDimension(); // 所有的优化变量总维度

        // 如果是SLAM问题, 则还要分别统计pose和landmark的维度, 后面会对它们进行排序
        if(_problem_type == SLAM_PROBLEM)
            AddOrderingSLAM(vertex.second);
    }

    if (_problem_type == SLAM_PROBLEM){
        // 这里要保证landmark在后, pose在前
        // 所以要把landmark的ordering加上pose的数量
        unsigned long all_pose_dimension = _ordering_poses;
        for(auto landmark_vertex: _idx_landmark_verticies){
            landmark_vertex.second->SetOrderingId(
                landmark_vertex.second->OrderingId() + all_pose_dimension
                );
        }
    }
}

bool Problem::Solve(int iterations)
{
    if (_edges.size() == 0 || _verticies.size() == 0)
        return false;

    // 统计优化变量的维度, 为构建H 矩阵做准备
    SetOrdering();
}
