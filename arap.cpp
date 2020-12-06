#include "arap.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

Mesh::Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
    : V(V), F(F),
      adjacency(edge_adjacency(V, F)) {
}

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

Eigen::SparseMatrix<double> cotangent_weights(const Mesh& mesh) {
    Eigen::SparseMatrix<double> weights(mesh.V.rows(), mesh.V.rows());

    for (int fid = 0; fid < mesh.F.rows(); fid++) {
	Eigen::Vector3i v(
	    mesh.F(fid, 0),
	    mesh.F(fid, 1),
	    mesh.F(fid, 2)
	    );

	Eigen::Matrix<double, 3, 3> edges;
	for (int j = 0; j < 3; j++) {
	    int k = (j + 1) % 3;

	    // edge vector between v(j) and v(k)
	    edges.row(j) = mesh.V.row(v(k)) - mesh.V.row(v(j));
	}

	double area_doubled = edges.row(0).cross(edges.row(1)).norm();
	double one_over_8area = 1.0 / (4 * area_doubled);

	for (int j = 0; j < 3; j++) {
	    int k = (j + 1) % 3;
	    int i = (j + 2) % 3;
	    
	    double d2 = edges.row(j).squaredNorm();

	    weights.coeffRef(v(j), v(k)) -= one_over_8area * d2 * .5;
	    weights.coeffRef(v(k), v(j)) -= one_over_8area * d2 * .5;
	    
	    weights.coeffRef(v(i), v(j)) += one_over_8area * d2;
	    weights.coeffRef(v(j), v(i)) += one_over_8area * d2;
	    
	    weights.coeffRef(v(i), v(k)) += one_over_8area * d2;
	    weights.coeffRef(v(k), v(i)) += one_over_8area * d2;
	}
    }

    return weights;
}


