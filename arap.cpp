#include "arap.hpp"

Eigen::SparseMatrix<bool> edge_adjacency(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
    Eigen::SparseMatrix<bool> adjacency(V.rows(), V.rows());
    
    for (int fid = 0; fid < F.rows(); fid++) {
	for (int j = 0; j < 3; j++) {
	    int k = (j + 1) % 3;
	    int v1 = F(fid, j);
	    int v2 = F(fid, k);
	    adjacency.coeffRef(v1, v2) = adjacency.coeffRef(v2, v1) = true;
	}
    }

    return adjacency;
}
