#include "arap.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

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

Eigen::SparseMatrix<double> cotangent_weights(const Mesh& mesh, std::vector<int> swizzle) {
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

	    int ci = swizzle[v(i)];
	    int cj = swizzle[v(j)];
	    int ck = swizzle[v(k)];

	    weights.coeffRef(cj, ck) -= one_over_8area * d2;
	    weights.coeffRef(ck, cj) -= one_over_8area * d2;
	    
	    weights.coeffRef(ci, cj) += one_over_8area * d2;
	    weights.coeffRef(cj, ci) += one_over_8area * d2;
	    
	    weights.coeffRef(ci, ck) += one_over_8area * d2;
	    weights.coeffRef(ck, ci) += one_over_8area * d2;
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

    Eigen::RowVector3d rval(0, 0, 0);
    for (Eigen::SparseMatrix<double>::InnerIterator it(weights, v); it; ++it) {
	Eigen::RowVector3d d = .5 * it.value() *
	    (mesh.V.row(it.col()) - mesh.V.row(it.row())) *
	    (rotations[it.row()] + rotations[it.col()]).transpose();
	// std::cout << "Adding " << d << "\n";
	// std::cout << "p[" << it.col() << "] - p[" << it.row() << "] = "
		  // << (mesh.V.row(it.col()) - mesh.V.row(it.row())) << "\n";

	rval += d;
    }

    // std::cout << "Returning " << rval << "\n";
    return rval;
}

std::vector<int> swizzle_from(int n, std::vector<int> fixed_indices) {
    std::vector<int> swizzled(n);
    
    int free_offset = 0;
    int fixed_offset = n - fixed_indices.size();

    int counter = 0;
    
    for (int fi : fixed_indices) {
	for (; counter < fi; counter++) {
	    swizzled[counter] = free_offset++;
	}
	swizzled[counter++] = fixed_offset++;
    }

    while (counter < n) {
	swizzled[counter++] = free_offset++;
    }

    return swizzled;
}

std::vector<int> reciprocal(std::vector<int> v) {
    std::vector<int> r(v.size());
    for (int i = 0; i < v.size(); i++) {
	r[v[i]] = i;
    }
    return r;
}

void system_init(LaplacianSystem& system, Mesh* mesh) {
    system.mesh = mesh;
    system.is_bound = false;
}

bool system_bind(LaplacianSystem& system, std::vector<int> fixed_indices) {
    system.V0 = system.mesh->V;
    system.is_bound = true;

    system.free_dimension = system.mesh->V.rows() - fixed_indices.size();
    
    system.swizzle = swizzle_from(system.mesh->V.rows(), fixed_indices);
    system.deswizzle = reciprocal(system.swizzle);
	
    system.cotangent_weights = cotangent_weights(*system.mesh, system.swizzle);

    Eigen::SparseMatrix<double> m = laplacian_matrix(system.cotangent_weights);
    system.laplacian_matrix = m.block(0, 0, system.free_dimension, system.free_dimension);
    system.fixed_constraint_matrix = m.block(0, system.free_dimension,
				      system.free_dimension, fixed_indices.size());
    
    system.rhs.resize(system.free_dimension, 3);

    system.solver.compute(system.laplacian_matrix);
    if (system.solver.info() != Eigen::Success) {
	return false;
    }
    
    return true;
}

bool system_iterate(LaplacianSystem& system) {
    std::vector<Eigen::Matrix3d> rotations(system.mesh->V.rows());
    for (int i = 0; i < system.mesh->V.rows(); i++) {
	rotations[i] = compute_best_rotation(*system.mesh,
					     system.cotangent_weights,
					     system.V0,
					     system.swizzle[i]);
	// std::cout << "rotation[" << i << "] = \n" << rotations[i] << "\n";
    }

    for (int i = 0; i < system.free_dimension; i++) {
	system.rhs.row(i) = laplacian_rhs(*system.mesh, system.cotangent_weights,
					  rotations, system.deswizzle[i]);
    }

    int n_fixed = system.mesh->V.rows() - system.free_dimension;
    Eigen::Matrix<double, Eigen::Dynamic, 3>
	V_fixed(n_fixed, 3);

    for (int i = 0; i < n_fixed; i++) {
	V_fixed.row(i) =
	    system.mesh->V.row(system.deswizzle[system.free_dimension + i]);
    }

    // std::cout << "REAL RHS:\n" << system.rhs << "\n\n";

    system.rhs -= system.fixed_constraint_matrix * V_fixed;

    // std::cout << "RHS AFTER compensation :\n" << system.rhs << "\n\n";
    
    Eigen::Matrix<double, Eigen::Dynamic, 3> solutions(system.free_dimension, 3);
    for (int i = 0; i < 3; i++) {
	solutions.col(i) = system.solver.solve(system.rhs.col(i));
	
	if (system.solver.info() != Eigen::Success) {
	    return false;
	}
	
    // 	std::cout << "MATRIX :\n"
    // 		  << system.laplacian_matrix
    // 		  << "\nRIGHT HAND SIDE :\n"
    // 		  << system.rhs.col(i)
    // 		  << "\nSOLUTION :\n"
    // 		  << solutions.col(i)
    // 		  << "\nMATRIX * SOLUTION :\n"
    // 		  << system.laplacian_matrix * solutions.col(i)
    // 		  << "\n---------\n\n\n";
    }

    // std::cout << "SOLUTIONS :\n" << solutions << "\n\n";
    
    // std::cout << "MATRIX * SOLUTIONS :\n" << system.laplacian_matrix * solutions << "\n\n";
    
    for (int i = 0; i < system.free_dimension; i++) {
	system.mesh->V.row(system.deswizzle[i]) = solutions.row(i);
    }

    return true;
}

void system_solve(LaplacianSystem& system) {
    // @TODO
    for (int i = 0; i < 10; i++) {
	system_iterate(system);
    }
}

