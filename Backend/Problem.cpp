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

void Problem::MakeHessian()
{
    unsigned long size = _ordering_generic;
    MatXX H = MatXX::Zero(size, size);
    VecX b = VecX::Zero(size);

    for(auto &edge: _edges){
        edge.second->ComputeResidual();
        edge.second->ComputeJacobians();

        auto jacobians = edge.second->Jacobians();
        auto verticies = edge.second->Verticies();
        for(int i = 0; i < verticies.size(); ++i){
            auto v_i = verticies[i];
            // 如果是fixed的话, 则在hessian中不需要添加这个vertex的信息, 也就是说这个vertex的雅克比为0
            if (v_i->IsFixed())
                continue;

            auto jacobian_i = jacobians[i];
            unsigned long index_i = v_i->OrderingId();
            unsigned long dim_i = v_i->LocalDimension();

            double drho;
            MatXX robust_info(edge.second->Information().rows(), edge.second->Information().cols());
            edge.second->RobustInfo(drho, robust_info);

            MatXX JtW = jacobian_i.transpose() * robust_info;
            for(int j = i; j < verticies.size(); ++j){
                auto v_j = verticies[j];
                if (v_j->IsFixed())
                    continue;
                auto jacobian_j = jacobians[j];
                unsigned long index_j = v_j->OrderingId();
                unsigned long dim_j = v_j->LocalDimension();

                MatXX hessian = JtW * jacobian_j;

                H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                if (j != i)
                    H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian;
            }

            b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose() * edge.second->Information() * edge.second->Residual();
        }
    }

    _Hessian = H;
    _b = b;

    if (_H_prior.rows() > 0){
        MatXX H_prior_tmp = _H_prior;
        VecX b_prior_tmp = _b_prior;

        // 遍历所有的POSE 顶点, 然后设置相应的先验维度为0, fix外参数
        // 同时我们要注意landmark 没有先验值
        // FIXME: 感觉没有什么值能满足这个要求.. 难道之前的帧都是Fixed? 并且新加了一帧, 这个大小还一样吗?
        for(auto vertex: _verticies){
            if(IsPoseVertex(vertex.second) && vertex.second->IsFixed()){
                int idx = vertex.second->OrderingId();
                int dim = vertex.second->LocalDimension();

                H_prior_tmp.block(idx, 0, dim, H_prior_tmp.cols()).setZero();
                H_prior_tmp.block(0, idx, H_prior_tmp.rows(), dim).setZero();
                b_prior_tmp.segment(idx, dim).setZero();
            }
        }

        _Hessian.topLeftCorner(_ordering_poses, _ordering_poses) += H_prior_tmp;
        _b.head(_ordering_poses) += b_prior_tmp;
    }

    // 初始值为0向量
    _delta_x = VecX::Zero(size);
}

void Problem::ComputeLambdaInitLM()
{
    _ni = 2.;
    _current_lambda = -1.;
    _current_chi = 0.0;

    for(auto edge: _edges)
        _current_chi += edge.second->RobustChi2();

    // FIXME: 还是同一个问题, 大小一样吗?
    if (_err_prior.rows() > 0)
        _current_chi += _err_prior.squaredNorm();

    _current_chi *= 0.5;
    _stop_threshold_LM = 1e-10 * _current_chi;

    double max_diagonal = 0;
    unsigned long size = _Hessian.cols();
    for(unsigned long i = 0; i < size; ++i)
        max_diagonal = std::max(fabs(_Hessian(i, i)), max_diagonal);

    max_diagonal = std::min(5e10, max_diagonal);
    double tau = 1e-5;
    _current_lambda = tau * max_diagonal;
}

void Problem::SolveLinearSystem()
{
    if (_problem_type == GENERIC_PROBLEM){
        MatXX H = _Hessian;
        for(int i = 0; i < _Hessian.cols(); ++i)
            H(i, i) += _current_lambda;
        _delta_x = H.ldlt().solve(_b);
    }
    else{
        // Schur marginalization
        int reserve_size = _ordering_poses;
        int marg_size = _ordering_landmarks;
        MatXX Hmm = _Hessian.block(reserve_size, reserve_size, marg_size, marg_size);
        MatXX Hpm = _Hessian.block(0, reserve_size, reserve_size, marg_size);
        MatXX Hmp = _Hessian.block(reserve_size, 0, marg_size, reserve_size);
        VecX bpp = _b.segment(0, reserve_size);
        VecX bmm = _b.segment(reserve_size, marg_size);
        MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));

        // Hmm为对角阵, 该矩阵的求逆直接为对角线块分别求逆, 如果是逆深度, 对角线块为1维的, 则直接为对角线的倒数
        for(auto landmark_vertex: _idx_landmark_verticies){
            int idx = landmark_vertex.second->OrderingId() - reserve_size;
            int size  = landmark_vertex.second->LocalDimension();
            Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
        }

        MatXX temp_H = Hpm * Hmm_inv;
        _Hpp_schur = _Hessian.block(0, 0, _ordering_poses, _ordering_poses) - temp_H * Hmp;
        _bpp_schur = bpp - temp_H * bmm;

        // solve Hpp * delta_x = bpp
        VecX delta_x_pp(VecX::Zero(reserve_size));

        for(unsigned long i = 0; i < _ordering_poses; ++i)
            _Hpp_schur(i, i) += _current_lambda; // LM method

        delta_x_pp = _Hpp_schur.ldlt().solve(_bpp_schur);
        _delta_x.head(reserve_size) = delta_x_pp;

        VecX delta_x_ll(marg_size);
        delta_x_ll = Hmm_inv * (bmm - Hmp * delta_x_pp);
        _delta_x.tail(marg_size) = delta_x_ll;
    }
}

void Problem::UpdateStates()
{
    for(auto vertex: _verticies){
        vertex.second->BackUpParameters(); // 保存上次的估计值

        unsigned long idx = vertex.second->OrderingId();
        unsigned long dim = vertex.second->LocalDimension();
        VecX delta = _delta_x.segment(idx, dim);
        vertex.second->Plus(delta);
    }

    // update prior
    if (_err_prior.rows() > 0){
        _b_prior_backup = _b_prior;
        _err_prior_backup = _err_prior;

        // update with first order Taylor
        _b_prior -= _H_prior * _delta_x.head(_ordering_poses);
        // FIXME: 这个15是什么东西?
        _err_prior = -_Jt_prior_inv * _b_prior.head(_ordering_poses - 15);
    }
}

bool Problem::IsGoodStepInLM()
{
    double scale = 0;
    // FIXME: 这个scale有什么用?
    scale = 0.5 * _delta_x.transpose() * (_current_lambda * _delta_x + _b);
    scale += 1e-6;

    // 在更新了状态后重新计算residuals
    double temp_chi = 0.0;
    for(auto edge: _edges){
        edge.second->ComputeResidual();
        temp_chi += edge.second->RobustChi2();
    }
    if (_err_prior.size() > 0)
        temp_chi += _err_prior.squaredNorm();
    temp_chi *= 0.5; // 1/2 * err^2

    double rho = (_current_chi - temp_chi) / scale;
    if (rho > 0 && std::isfinite(temp_chi)) { // 误差在下降, 则使用高斯牛顿法
        double alpha = 1. - pow((2 * rho - 1), 3);
        alpha = std::min(alpha, 2. / 3.);
        double scale_factor = std::max(1. / 3., alpha);
        _current_lambda *= scale_factor;
        _ni = 2;
        _current_chi = temp_chi;
        return true;
    }
    // 如果误差上升, 则趋近于梯度下降法
    else{
        _current_lambda *= _ni;
        _ni *= 2;
        return false;
    }
}

void Problem::RollbackStates()
{
    for(auto vertex: _verticies)
        vertex.second->RollBackParameters();
    if (_err_prior.rows() > 0){
        _b_prior = _b_prior_backup;
        _err_prior = _err_prior_backup;
    }
}

// 这里iterations的含义是成功迭代10次
// FIXME: 感觉这个条件很僵硬
bool Problem::Solve(int iterations)
{
    if (_edges.size() == 0 || _verticies.size() == 0)
        return false;

    // 统计优化变量的维度, 为构建H 矩阵做准备
    SetOrdering();
    // 遍历edge, 构建H矩阵
    MakeHessian();
    // LM初始化
    ComputeLambdaInitLM();

    bool stop = false;
    int iter = 0;
    double last_chi = 1e20;
    while(!stop && iter < iterations){
        std::cout << "iter: " << iter << " , chi= " << _current_chi << " , Lambda= " << _current_lambda << std::endl;
        bool one_step_success = false;
        int false_cnt = 0;

        while(!one_step_success && false_cnt) // 不断尝试Lambda, 直到成功迭代一步
        {
            // 求解线性方程组
            SolveLinearSystem();
            // 更新状态量
            UpdateStates();
            // 判断当前步是否可行以及LM的lambda怎么更新, chi2也计算一次
            one_step_success = IsGoodStepInLM();

            if (one_step_success){
                MakeHessian();
                false_cnt = 0;
            }
            else{
                false_cnt++;
                RollbackStates();// 状态没下降, 回滚
            }
        }

        iter++;
        if (last_chi - _current_chi < 1e-5){
            std::cout << "the error is too small." << std::endl;
            stop = true;
        }
        last_chi = _current_chi;
    }

    return true;
}
