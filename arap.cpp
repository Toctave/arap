#include "arap.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <algorithm> // std::sort

Eigen::SparseMatrix<double> cotangent_weights(const Mesh& mesh, const std::vector<Eigen::Index>& swizzle) {
    Eigen::SparseMatrix<double> weights(mesh.V.rows(), mesh.V.rows());
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(18 * mesh.F.rows());

    for (int fid = 0; fid < mesh.F.rows(); fid++) {
	Eigen::Vector3i v = mesh.F.row(fid);

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

Eigen::Matrix3d compute_best_rotation(const LaplacianSystem& system, int r) {
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();

    for (Eigen::SparseMatrix<double>::InnerIterator it(system.cotangent_weights, r); it; ++it) {
	Eigen::Index v_idx[2] = {
	    system.deswizzle[it.col()],
	    system.deswizzle[it.row()]
	};

	Eigen::Vector3d e = system.mesh->V.row(v_idx[0]) - system.mesh->V.row(v_idx[1]);
	Eigen::Vector3d e0 = system.V0.row(v_idx[0]) - system.V0.row(v_idx[1]);

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

std::vector<Eigen::Index> swizzle_from(int n, const std::vector<FixedVertex>& fixed_vertices) {
    std::vector<Eigen::Index> swizzled(n);
    
    size_t free_offset = 0;
    size_t fixed_offset = n - fixed_vertices.size();

    size_t swizzled_offset = 0;
    
    for (int i = 0; i < fixed_vertices.size(); i++) {
	for (; swizzled_offset < fixed_vertices[i].index; swizzled_offset++) {
	    swizzled[swizzled_offset] = free_offset++;
	}
	swizzled[swizzled_offset] = fixed_offset++;
	swizzled_offset++;
    }
    
    for (; swizzled_offset < n; swizzled_offset++) {
	swizzled[swizzled_offset] = free_offset++;
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

bool system_bind(LaplacianSystem& system, const std::vector<FixedVertex>& fixed_vertices) {
    system.V0 = system.mesh->V;
    system.is_bound = true;

    system.free_dimension = system.mesh->V.rows() - fixed_vertices.size();
    
    system.swizzle = swizzle_from(system.mesh->V.rows(), fixed_vertices);
    system.deswizzle = reciprocal(system.swizzle);
	
    system.cotangent_weights = cotangent_weights(*system.mesh, system.swizzle);

    Eigen::SparseMatrix<double> m = laplacian_matrix(system.cotangent_weights);
    system.laplacian_matrix = m.block(0, 0, system.free_dimension, system.free_dimension);
    system.fixed_constraint_matrix = m.block(0, system.free_dimension,
					     system.free_dimension, fixed_vertices.size());
    
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
	rotations[i] = compute_best_rotation(system, i);
    }

    /* --- Fill system's right hand side --- */
    
    system.rhs.setZero();

    for (int v = 0; v < system.free_dimension; v++) {
	for (Eigen::SparseMatrix<double>::InnerIterator
		 it(system.cotangent_weights, v);
	     it;
	     ++it) {
	    Eigen::Index v_idx[2] = {
		system.deswizzle[it.col()],
		system.deswizzle[it.row()],
	    };
	    
	    Eigen::RowVector3d d = .5 * it.value() *
		(system.V0.row(v_idx[0]) - system.V0.row(v_idx[1])) *
		(rotations[it.row()] + rotations[it.col()]).transpose();

	    system.rhs.row(v) += d;
	}
    }

    int n_fixed = system.mesh->V.rows() - system.free_dimension;
    Eigen::Matrix<double, Eigen::Dynamic, 3>
	V_fixed(n_fixed, 3);

    for (int i = 0; i < n_fixed; i++) {
	V_fixed.row(i) =
	    system.mesh->V.row(system.deswizzle[system.free_dimension + i]);
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
	system.mesh->V.row(system.deswizzle[i])
	    = solutions.row(i);
    }

    return true;
}

void system_solve(LaplacianSystem& system, int iterations) {
    for (int i = 0; i < iterations; i++) {
	system_iterate(system);
    }
}
