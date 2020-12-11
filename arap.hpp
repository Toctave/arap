#pragma once
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

struct Mesh {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
};

struct FixedVertex {
    Eigen::Index index;
    size_t group;
};

struct LaplacianSystem {
    Mesh* mesh;
    bool is_bound;
    int free_dimension;
    std::vector<Eigen::Index> swizzle;
    std::vector<Eigen::Index> deswizzle;
    
    Eigen::MatrixXd V0;
    Eigen::SparseMatrix<double> laplacian_matrix;
    Eigen::SparseMatrix<double> fixed_constraint_matrix;
    Eigen::SparseMatrix<double> cotangent_weights;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    Eigen::Matrix<double, Eigen::Dynamic, 3> rhs;
};

std::vector<Eigen::Index> swizzle_from(int n, const std::vector<FixedVertex>& fixed_indices);

void system_init(LaplacianSystem& system, Mesh* mesh);
bool system_bind(LaplacianSystem& system, const std::vector<FixedVertex>& fixed_vertices);
void system_solve(LaplacianSystem& system, int iterations);
bool system_iterate(LaplacianSystem& system);


