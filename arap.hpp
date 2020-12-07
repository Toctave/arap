#pragma once
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

struct Mesh {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
};

struct LaplacianSystem {
    Mesh* mesh;
    bool is_bound;
    int free_dimension;
    std::vector<int> swizzle;
    std::vector<int> deswizzle;
    
    Eigen::MatrixXd V0;
    Eigen::SparseMatrix<double> laplacian_matrix;
    Eigen::SparseMatrix<double> fixed_constraint_matrix;
    Eigen::SparseMatrix<double> cotangent_weights;

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    Eigen::Matrix<double, Eigen::Dynamic, 3> rhs;
};

std::vector<int> swizzle_from(int n, std::vector<int> fixed_indices);

Eigen::SparseMatrix<bool> edge_adjacency(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
// Eigen::SparseMatrix<double> cotangent_weights(const Mesh& mesh);
Eigen::Matrix3d compute_best_rotation(
    const Mesh& mesh, const Eigen::SparseMatrix<double>& weights,
    const Eigen::MatrixXd& V0, int v);


void system_init(LaplacianSystem& system, Mesh* mesh);
bool system_bind(LaplacianSystem& system, std::vector<int> fixed_indices);
void system_solve(LaplacianSystem& system);
bool system_iterate(LaplacianSystem& system);


