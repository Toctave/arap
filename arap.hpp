#pragma once
#include <Eigen/SparseCore>

struct Mesh {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
};

struct LaplacianSystem {
    Mesh* mesh;
    bool is_bound;
    int free_dimension;
    std::vector<int> fixed_indices;
    std::vector<int> swizzle;
    std::vector<int> deswizzle;
    
    Eigen::MatrixXd V0;
    Eigen::SparseMatrix<double> matrix;
    Eigen::SparseMatrix<double> cotangent_weights;

    Eigen::Matrix<double, Eigen::Dynamic, 3> rhs;
};

std::vector<int> swizzle_from(int n, std::vector<int> fixed_indices);

Eigen::SparseMatrix<bool> edge_adjacency(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
// Eigen::SparseMatrix<double> cotangent_weights(const Mesh& mesh);
Eigen::Matrix3d compute_best_rotation(
    const Mesh& mesh, const Eigen::SparseMatrix<double>& weights,
    const Eigen::MatrixXd& V0, int v);

