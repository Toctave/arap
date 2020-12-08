#include "arap.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

Eigen::SparseMatrix<double> cotangent_weights(const Mesh& mesh, const std::vector<Eigen::Index>& swizzle) {
    Eigen::SparseMatrix<double> weights(mesh.V.rows(), mesh.V.rows());
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(6 * mesh.F.rows());

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

	    int ci = v(i);
	    int cj = v(j);
	    int ck = v(k);

	    double contribution = one_over_8area * d2;

	    triplets.push_back(Eigen::Triplet<double>(cj, ck, -contribution));
	    triplets.push_back(Eigen::Triplet<double>(ck, cj, -contribution));
	    
	    triplets.push_back(Eigen::Triplet<double>(ci, cj, contribution));
	    triplets.push_back(Eigen::Triplet<double>(cj, ci, contribution));
	    
	    triplets.push_back(Eigen::Triplet<double>(ci, ck, contribution));
	    triplets.push_back(Eigen::Triplet<double>(ck, ci, contribution));
	}
    }

    weights.setFromTriplets(triplets.begin(), triplets.end());

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

	// std::cout << "e[" << it.row() << " -> " << it.col() << "] = " << e.transpose() << "\n";
	
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

    assert(fabs(rot.determinant() - 1.0) < 1e-3);

    return rot;
}

std::vector<Eigen::Index> swizzle_from(int n, const std::vector<Eigen::Index>& fixed_indices) {
    std::vector<Eigen::Index> swizzled(n);
    
    int free_offset = 0;
    int fixed_offset = n - fixed_indices.size();

    int counter = 0;

    for (int fi : fixed_indices) {
	for (; counter < fi; counter++) {
	    swizzled[counter] = free_offset++;
	}
	free_offset++;
    }

    //Completing free indices
    while (counter < fixed_offset) {
	swizzled[counter++] = free_offset++;
    }

    //Completing fixed indices
    for (int i = 0; i < fixed_indices.size(); i++) {
        swizzled[fixed_offset + i] = fixed_indices[i];
    }

    return swizzled;
}

std::vector<Eigen::Index> reciprocal(const std::vector<Eigen::Index>& v) {
    std::vector<Eigen::Index> r(v.size());
    for (int i = 0; i < v.size(); i++) {
	r[v[i]] = i;
    }
    return r;
}

void system_init(LaplacianSystem& system, Mesh* mesh) {
    system.mesh = mesh;
    system.is_bound = false;
}

bool system_bind(LaplacianSystem& system, const std::vector<Eigen::Index>& fixed_indices) {
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

    // std::cout << "Cotangent weights :\n" << system.cotangent_weights << "\n";
    // std::cout << "Laplacian matrix :\n" << system.laplacian_matrix << "\n";
    // std::cout << "A matrix :\n" << system.fixed_constraint_matrix << "\n";
    
    return true;
}

bool system_iterate(LaplacianSystem& system) {
    /* --- Compute approximate rotations --- */
    
    std::vector<Eigen::Matrix3d> rotations(system.mesh->V.rows());
    for (int i = 0; i < system.mesh->V.rows(); i++) {
	rotations[i] = compute_best_rotation(*system.mesh,
					     system.cotangent_weights,
					     system.V0,
					     i);
	// std::cout << "rotation[" << i << "] = \n" << rotations[i] << "\n";
    }

    /* --- Fill system's right hand side --- */
    
    system.rhs.setZero();

    for (int v = 0; v < system.free_dimension; v++) {
	for (Eigen::SparseMatrix<double>::InnerIterator
		 it(system.cotangent_weights, v);
	     it;
	     ++it) {
	    Eigen::RowVector3d d = .5 * it.value() *
		(system.V0.row(it.col()) - system.V0.row(it.row())) *
		(rotations[it.row()] + rotations[it.col()]); //.transpose();
	    // std::cout << "p[" << it.col() << "] - p[" << it.row() << "] = "
	    // << (mesh.V.row(it.col()) - mesh.V.row(it.row())) << "\n";
	    system.rhs.row(v) += d;
	}
    }

    int n_fixed = system.mesh->V.rows() - system.free_dimension;
    Eigen::Matrix<double, Eigen::Dynamic, 3>
	V_fixed(n_fixed, 3);

    for (int i = 0; i < n_fixed; i++) {
	V_fixed.row(i) =
	    system.V0.row(system.free_dimension + i);
    }

    system.rhs -= system.fixed_constraint_matrix * V_fixed;
    
    Eigen::Matrix<double, Eigen::Dynamic, 3> solutions(system.free_dimension, 3);
    for (int i = 0; i < 3; i++) {
	solutions.col(i) = system.solver.solve(system.rhs.col(i));
	
	if (system.solver.info() != Eigen::Success) {
	    return false;
	}
    }
    assert((system.laplacian_matrix * solutions
	    - system.rhs).norm() < 1e-3);

    for (int i = 0; i < system.free_dimension; i++) {
	system.mesh->V.row(i) = solutions.row(i);
    }

    // std::cout << "Solutions :\n" << solutions << "\n\n";

    return true;
}

void system_solve(LaplacianSystem& system) {
    // @TODO
    for (int i = 0; i < 10; i++) {
	system_iterate(system);
    }
}

