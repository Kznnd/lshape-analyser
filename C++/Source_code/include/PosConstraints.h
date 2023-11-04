/*
 * PosConstraints.h
 *
 *  Created on: August 30, 2023
 *      Author: Jialong Zhang
 * Reference:
 * Aigerman N, Lipman Y.Orbifold tutte embeddings[J].ACM Trans.Graph.,
 *	2015, 34(6) : 190 : 1 - 190 : 12.
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

class PosConstraints
{
public:
	PosConstraints() {};
	~PosConstraints() {};
	Eigen::SparseMatrix<double> A;
	Eigen::VectorXd b;

	void addConstraint(size_t ind, int w, Eigen::Vector2d rhs);
	void addLineConstraint(size_t ind, Eigen::Vector2d n, double offset);
};