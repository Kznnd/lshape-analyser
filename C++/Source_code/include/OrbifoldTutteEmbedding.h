/*
 * OrbifoldTutteEmbedding.h
 *
 *  Created on: August 30, 2023
 *      Author: Jialong Zhang
 * 
 * Reference:
 * Aigerman N, Lipman Y.Orbifold tutte embeddings[J].ACM Trans.Graph.,
 *	2015, 34(6) : 190 : 1 - 190 : 12.
 */

#pragma once

#include <Eigen/Sparse>

#include "LShapeAnalyserFunction.h"

class OrbifoldTutteEmbedding
{
public:
	bool disk_map(Eigen::MatrixX3d V1, Eigen::MatrixX3i T1, Eigen::VectorXi b1, Eigen::MatrixX3d V2, 
		Eigen::MatrixX3i T2, Eigen::VectorXi b2, Eigen::VectorXi sc1, Eigen::VectorXi sc2, Eigen::MatrixX3d& verticesMapped);

	void map_disks(Eigen::MatrixX3d V1, Eigen::MatrixX3i T1, Eigen::VectorXi sc1, Eigen::MatrixX3d V2,
		Eigen::MatrixX3i T2, Eigen::VectorXi sc2, Eigen::VectorXi b1, Eigen::VectorXi b2,
		Eigen::SparseMatrix<double>& BC_1to2);

	void compute_map_from_disk_embeddings(Eigen::MatrixX2d V_source, Eigen::MatrixX3i T_source,
		Eigen::MatrixX2d V_target, Eigen::MatrixX3i T_target, std::vector<Eigen::VectorXi> boundary_source,
		std::vector<Eigen::VectorXi> boundary_target, Eigen::SparseMatrix<double>& BC);

	void flatten_disk(Eigen::MatrixX3d V, Eigen::MatrixX3i T, Eigen::VectorXi inds, Eigen::MatrixX2d& V_flat,
		std::vector<Eigen::VectorXi>& segs, Eigen::VectorXi b);

	void flatten(bool tri_or_square, Eigen::VectorXi all_binds, Eigen::VectorXi p, Eigen::MatrixX3d V,
		Eigen::MatrixX3i T, Eigen::MatrixX2d& V_flat);

	void computeFlattening(Eigen::SparseMatrix<double> A, Eigen::VectorXd b,
		Eigen::SparseMatrix<double> L, Eigen::VectorXd& x);

	Eigen::SparseMatrix<double> cotmatrix(Eigen::MatrixX3d V, Eigen::MatrixX3i F);

	Eigen::SparseMatrix<double> mean_value_laplacian(Eigen::MatrixX3d V, Eigen::MatrixX3i F);

	void cal_barycentric(Point_2d p1, Point_2d p2, Point_2d p3, Point_2d p, double& alpha, double& beta, double& gamma);
};