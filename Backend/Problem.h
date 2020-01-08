//
// Created by liu on 20-1-6.
//

#ifndef VIO_EXAMPLE_PROBLEM_H
#define VIO_EXAMPLE_PROBLEM_H

#include <memory>
#include <map>
#include <unordered_map>
#include <iostream>
#include <algorithm>

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
    bool RemoveVertex(std::shared_ptr<Vertex> vertex);

    bool AddEdge(std::shared_ptr<Edge> edge);
    bool RemoveEdge(std::shared_ptr<Edge> edge);

    bool Solve(int iterations = 10);

    MatXX GetHessianPrior(){ return _H_prior;}
    VecX GetbPrior(){ return _b_prior;}
    VecX GetErrPrior(){ return _err_prior;}
    MatXX GetJtPrior(){ return _Jt_prior_inv;}

    void SetHessianPrior(const MatXX& H){_H_prior = H;}
    void SetbPrior(const VecX& b){_b_prior = b;}
    void SetErrPrior(const VecX& b){_err_prior = b;}
    void SetJtPrior(const MatXX& J){_Jt_prior_inv = J;}
    void ExtendHessiansPriorSize(int dim);

private:
    // 设置各顶点的ordering_index
    void SetOrdering();

    // 新增顶点后, 需要调整几个Hessian的大小.
    void ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v);

    // 检测一个顶点是否为Pose
    bool IsPoseVertex(std::shared_ptr<Vertex> v){
        std::string type = v->TypeInfo();
        return type == std::string("VertexPose") || type == std::string("VertexSpeedBias");
    }

    // 检测一个顶点是否为landmark
    bool IsLandmarkVertex(std::shared_ptr<Vertex> v){
        std::string type = v->TypeInfo();
        return type == std::string("VertexPointXYZ") || type == std::string("VertexInverseDepth");
    }

    // 获取某个顶点连接的边
    std::vector<std::shared_ptr<Edge> > GetConnectedEdges(std::shared_ptr<Vertex> vertex);

    // set ordering for new vertex in slam problem
    void AddOrderingSLAM(std::shared_ptr<Vertex> v);

    void MakeHessian();

    void ComputeLambdaInitLM();

    void SolveLinearSystem();

    void UpdateStates();

    // LM算法中用于判断lambda 在上次迭代中是否可以, 以及lambda怎么缩放.
    bool IsGoodStepInLM();

    void RollbackStates();

private:
    double _current_lambda;
    double _current_chi;
    double _stop_threshold_LM; // LM迭代退出阈值条件
    double _ni; // 控制Lambda缩放大小

    ProblemType _problem_type;

    // all verticies
    HashVertex _verticies;

    // all edges
    HashEdge _edges;

    // 由vertex id 查询edge
    HashVertexIdToEdge _vertexToEdge;

    // verticies need to marg
    HashVertex _verticies_marg;

    // 整个信息矩阵
    MatXX _Hessian;
    VecX _b;
    VecX _delta_x;

    // 先验部分信息
    MatXX _H_prior;
    VecX _b_prior;
    VecX _b_prior_backup;
    VecX _err_prior_backup;

    MatXX _Jt_prior_inv;
    VecX _err_prior;

    // SBA的Pose部分
    MatXX _Hpp_schur;
    VecX _bpp_schur;
    // Hessian的landmark和pose部分
    MatXX _Hpp;
    VecX _bpp;
    MatXX _Hll;
    VecX _bll;

    // ordering related
    unsigned long _ordering_poses = 0; // poses的大小
    unsigned long _ordering_landmarks = 0; // landmarks的大小
    unsigned long _ordering_generic = 0; // poses+landmarks的总大小
    // id为vertex的id
    std::map<unsigned long, std::shared_ptr<Vertex> > _idx_pose_verticies; // 以ordering排序的pose顶点
    // id为landmark的id
    std::map<unsigned long, std::shared_ptr<Vertex> > _idx_landmark_verticies; // 以ordering排序的landmark顶点
};


#endif //VIO_EXAMPLE_PROBLEM_H
