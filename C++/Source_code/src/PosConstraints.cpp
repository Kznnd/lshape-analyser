/*
 * PosConstraints.cpp
 *
 *  Created on: August 30, 2023
 *      Author: Jialong Zhang
 * Reference:
 * Aigerman N, Lipman Y.Orbifold tutte embeddings[J].ACM Trans.Graph.,
 *	2015, 34(6) : 190 : 1 - 190 : 12.
 */

#include "PosConstraints.h"

void PosConstraints::addConstraint(size_t ind, int w, Eigen::Vector2d rhs) {
	if (A.cols() < (ind + 1) * 2)
		A.conservativeResize(A.rows() + 2, (ind + 1) * 2);
	else
		A.conservativeResize(A.rows() + 2, A.cols());
	A.insert(A.rows() - 2, ind * 2) = w;
	A.insert(A.rows() - 1, ind * 2 + 1) = w;

	b.conservativeResize(b.size() + 2);
	b[b.size() - 2] = rhs[0];
	b[b.size() - 1] = rhs[1];
}


void PosConstraints::addLineConstraint(size_t ind, Eigen::Vector2d n, double offset) {
	if (A.cols() < (ind + 1) * 2)
		A.conservativeResize(A.rows() + 1, (ind + 1) * 2);
	else
		A.conservativeResize(A.rows() + 1, A.cols());
	A.insert(A.rows() - 1, ind * 2) = n[0];
	A.insert(A.rows() - 1, ind * 2 + 1) = n[1];

	b.conservativeResize(b.size() + 1);
	b[b.size() - 1] = offset;
}
