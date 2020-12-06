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

	    weights.coeffRef(v(j), v(k)) -= one_over_8area * d2;
	    weights.coeffRef(v(k), v(j)) -= one_over_8area * d2;
	    
	    weights.coeffRef(v(i), v(j)) += one_over_8area * d2;
	    weights.coeffRef(v(j), v(i)) += one_over_8area * d2;
	    
	    weights.coeffRef(v(i), v(k)) += one_over_8area * d2;
	    weights.coeffRef(v(k), v(i)) += one_over_8area * d2;
	}
    }

    return weights;
}

Eigen::SparseMatrix<double> laplacian_matrix(
    const Eigen::SparseMatrix<double>& weights) {
    
    Eigen::SparseMatrix<double> mat = -weights;

    for (int k=0; k < mat.outerSize(); ++k) {
	double colsum = 0;
	for (Eigen::SparseMatrix<double>::InnerIterator it(mat,k); it; ++it) {
	    colsum += it.value();
	}
	mat.coeffRef(k, k) = -colsum;
    }

    return mat;
}

Eigen::Matrix3d compute_best_rotation(
    const Mesh& mesh, const Eigen::SparseMatrix<double>& weights,
    const Eigen::MatrixXd& V0, int v) {
    
    Eigen::Matrix3d cov;

    for (Eigen::SparseMatrix<double>::InnerIterator it(weights, v); it; ++it) {
	Eigen::Vector3d e = mesh.V.row(it.col()) - mesh.V.row(it.row());
	// @opti : e0 could be precomputed
	Eigen::Vector3d e0 = V0.row(it.col()) - V0.row(it.row());
	cov += it.value() * e0 * e.transpose();
    }
    
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3d um = svd.matrixU();
    Eigen::Matrix3d vm = svd.matrixV();
    Eigen::Matrix3d rot = vm * um.transpose();

    if (rot.determinant() < 0) {
	um.col(2) *= -1;
	rot = vm * um.transpose();
    }
    
    return rot;
}

Eigen::RowVector3d laplacian_rhs(
    const Mesh& mesh, const Eigen::SparseMatrix<double>& weights,
    std::vector<Eigen::Matrix3d> rotations, int v) {

    Eigen::RowVector3d rval;
    for (Eigen::SparseMatrix<double>::InnerIterator it(weights, v); it; ++it) {
	rval += .5 * it.value() *
	    (mesh.V.row(it.col()) - mesh.V.row(it.row())) *
	    (rotations[it.row()] * rotations[it.col()]).transpose();
    }

    return rval;
}

void setup_laplacian_system(const Mesh& mesh, const Eigen::MatrixXd& V0) {
    Eigen::SparseMatrix<double> weights = cotangent_weights(mesh);
    Eigen::SparseMatrix<double> lapmat = laplacian_matrix(weights);

    std::vector<Eigen::Matrix3d> rotations(mesh.V.rows());
    Eigen::Matrix<double, Eigen::Dynamic, 3> rhs;
    rhs.resize(mesh.V.rows(), 3);

    for (int i = 0; i < mesh.V.rows(); i++) {
	rotations[i] = compute_best_rotation(mesh, weights, V0, i);
    }
    
    for (int i = 0; i < mesh.V.rows(); i++) {
	rhs.row(i) = laplacian_rhs(mesh, weights, rotations, i);
    }

    // now V' is such that :
    // lapmat * V' = rhs
}
