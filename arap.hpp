#pragma once
#include <Eigen/SparseCore>

Eigen::SparseMatrix<bool> edge_adjacency(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
