#pragma once
#include <Eigen/SparseCore>

struct Mesh {
    Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
    
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::SparseMatrix<bool> adjacency;
};

Eigen::SparseMatrix<bool> edge_adjacency(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
Eigen::SparseMatrix<double> cotangent_weights(const Mesh& mesh);
Eigen::SparseMatrix<double> laplacian_matrix(const Mesh& mesh);
    
