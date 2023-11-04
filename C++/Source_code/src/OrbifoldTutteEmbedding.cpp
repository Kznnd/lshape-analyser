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

#include <iostream>
#include <fstream>

#include <QMessageBox>

#include <Eigen/Sparse>

#include "PosConstraints.h"
#include "OrbifoldTutteEmbedding.h"
#include "LShapeAnalyserFunction.h"


bool isfail=false;


void OrbifoldTutteEmbedding::cal_barycentric(Point_2d p1, Point_2d p2, Point_2d p3, Point_2d p, double& alpha, double& beta, double& gamma) {
	double denominator = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y);

	double numeratorAlpha = (p2.x - p.x) * (p3.y - p.y) - (p3.x - p.x) * (p2.y - p.y);
	alpha = numeratorAlpha / denominator;

	double numeratorBeta = (p3.x - p.x) * (p1.y - p.y) - (p1.x - p.x) * (p3.y - p.y);
	beta = numeratorBeta / denominator;

	gamma = 1 - alpha - beta;
}


Eigen::SparseMatrix<double> OrbifoldTutteEmbedding::mean_value_laplacian(Eigen::MatrixX3d V, Eigen::MatrixX3i F) {
	int n = V.rows();
	Eigen::MatrixX3d l(F.rows(), 3);
	Eigen::VectorXd l1(F.rows()), l2(F.rows()), l3(F.rows()), s(F.rows()), dblA(F.rows()), 
		halftan12(F.rows()), halftan23(F.rows()), halftan31(F.rows()), Lv(F.rows() * 6);
	
	for (int i = 0; i < F.rows(); i++) {
		l1[i] = sqrt((V.row(F(i, 1)) - V.row(F(i, 2))).dot(V.row(F(i, 1)) - V.row(F(i, 2))));
		l2[i] = sqrt((V.row(F(i, 2)) - V.row(F(i, 0))).dot(V.row(F(i, 2)) - V.row(F(i, 0))));
		l3[i] = sqrt((V.row(F(i, 0)) - V.row(F(i, 1))).dot(V.row(F(i, 0)) - V.row(F(i, 1))));
	}
	l << l1, l2, l3;

	Eigen::VectorXi i1(F.rows()), i2(F.rows()), i3(F.rows()), Li(F.rows() * 6), Lj(F.rows() * 6);
	i1 << F.col(0);
	i2 << F.col(1);
	i3 << F.col(2);

	s = (l1 + l2 + l3) * 0.5;

	dblA = 2 * (s.cwiseProduct(s - l1).cwiseProduct(s - l2).cwiseProduct(s - l3)).cwiseSqrt();

	halftan12 = 2 * dblA.cwiseQuotient(2 * l1.cwiseProduct(l2) + l1.cwiseProduct(l1) + l2.cwiseProduct(l2) - l3.cwiseProduct(l3));
	halftan23 = 2 * dblA.cwiseQuotient(2 * l2.cwiseProduct(l3) + l2.cwiseProduct(l2) + l3.cwiseProduct(l3) - l1.cwiseProduct(l1));
	halftan31 = 2 * dblA.cwiseQuotient(2 * l3.cwiseProduct(l1) + l3.cwiseProduct(l3) + l1.cwiseProduct(l1) - l2.cwiseProduct(l2));

	Li << i1,
		i1,
		i2,
		i2,
		i3,
		i3;
	Lj << i2,
		i3,
		i3,
		i1,
		i1,
		i2;
	Lv << halftan23.cwiseQuotient(l3),
		halftan23.cwiseQuotient(l2),
		halftan31.cwiseQuotient(l1),
		halftan31.cwiseQuotient(l3),
		halftan12.cwiseQuotient(l2),
		halftan12.cwiseQuotient(l1);

	Eigen::SparseMatrix<double> L(n, n);
	for (int i = 0; i < F.rows() * 6; i++) 
			L.coeffRef(Li[i], Lj[i]) = L.coeff(Li[i], Lj[i]) + Lv[i];

	for (int i = 0; i < n; i++) 
		L.coeffRef(i, i) = L.coeff(i, i) - L.row(i).sum();
	
	return L;
}


Eigen::SparseMatrix<double> OrbifoldTutteEmbedding::cotmatrix(Eigen::MatrixX3d V, Eigen::MatrixX3i F) {
	Eigen::VectorXi i1(F.rows()), i2(F.rows()), i3(F.rows());
	Eigen::MatrixX3d v1(F.rows(), 3), v2(F.rows(), 3), v3(F.rows(), 3), n(F.rows(), 3);
	for (int k = 0; k < F.rows(); k++) {
		i1[k] = F(k, 0);
		i2[k] = F(k, 1);
		i3[k] = F(k, 2);
		for (int l = 0; l < 3; l++) {
			v1(k, l) = V(i3[k], l) - V(i2[k], l);
			v2(k, l) = V(i1[k], l) - V(i3[k], l);
			v3(k, l) = V(i2[k], l) - V(i1[k], l);
		}
	}
	
	for (int k = 0; k < F.rows(); k++) {
		n(k, 0) = v1(k, 1) * v2(k, 2) - v1(k, 2) * v2(k, 1);
		n(k, 1) = -v1(k, 0) * v2(k, 2) + v1(k, 2) * v2(k, 0);
		n(k, 2) = v1(k, 0) * v2(k, 1) - v1(k, 1) * v2(k, 0);
	}

	Eigen::VectorXd dblA(F.rows()), cot12(F.rows()), cot23(F.rows()), cot31(F.rows()), 
		diag1(F.rows()), diag2(F.rows()), diag3(F.rows());
	dblA = n.array().square().rowwise().sum().sqrt();

	cot12 = -v1.cwiseProduct(v2).rowwise().sum().cwiseQuotient(dblA) / 2;
	cot23 = -v2.cwiseProduct(v3).rowwise().sum().cwiseQuotient(dblA) / 2;
	cot31 = -v3.cwiseProduct(v1).rowwise().sum().cwiseQuotient(dblA) / 2;

	diag1 = -cot12 - cot31;
	diag2 = -cot12 - cot23;
	diag3 = -cot31 - cot23;

	Eigen::VectorXi i(F.rows() * 9), j(F.rows() * 9);
	Eigen::VectorXd v(F.rows() * 9);
	i << i1,
		i2,
		i2,
		i3,
		i3,
		i1,
		i1,
		i2,
		i3;
	j << i2,
		i1,
		i3,
		i2,
		i1,
		i3,
		i1,
		i2,
		i3;
	v << cot12,
		cot12,
		cot23, 
		cot23,
		cot31, 
		cot31,
		diag1,
		diag2, 
		diag3;
	Eigen::SparseMatrix<double> L(V.rows(), V.rows());
	for (int n = 0; n < F.rows() * 9; n++) 
			L.coeffRef(i[n], j[n]) = L.coeff(i[n], j[n]) + v[n];
	L.makeCompressed();
	double trainMesh = std::numeric_limits<double>::max();
	for (int k = 0; k < L.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
			int row = it.row();
			int col = it.col();
			if (row > col && it.value() < trainMesh) {//Traverse the lower triangular matrix
				trainMesh = it.value();
			}
		}
	}
	if (trainMesh < 0) {
		for (int k = 0; k < L.outerSize(); ++k) {
			for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
				int row = it.row();
				int col = it.col();
				if (it.value() < 0) 
					L.coeffRef(row, col) = 0;
				if (row == col) 
					L.coeffRef(row, col) = 0;
			}
		}
		L.makeCompressed();
		//now fix the laplacian
		for (int k = 0; k < L.outerSize(); ++k) {
			for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
				int row = it.row();
				int col = it.col();
				double value = it.value();
				if(row!=col)
					L.coeffRef(row, row) = L.coeff(row, row) - value;
			}
		}
	}
	L.makeCompressed();
	return L;
}


void OrbifoldTutteEmbedding::computeFlattening(Eigen::SparseMatrix<double> A, Eigen::VectorXd b,
	Eigen::SparseMatrix<double> L, Eigen::VectorXd& x) {
	int n_vars = L.rows();
	int n_eq = A.rows();
	Eigen::SparseMatrix<double> M(L.rows() + A.rows(), L.cols() + A.rows());
	
	for (int i = 0; i < L.outerSize(); ++i) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(L, i); it; ++it) {
			int row = it.row();
			int col = it.col();
			double value = it.value();
			M.insert(row, col) = value;
		}
	}
	A.makeCompressed();
	for (int i = 0; i < A.outerSize(); ++i) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
			int row = it.col();
			int col = it.row() + L.cols();
			double value = it.value();
			M.insert(row, col) = value;
		}
	}

	for (int i = 0; i < A.outerSize(); ++i) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
			int row = it.row() + L.rows();
			int col = it.col();
			double value = it.value();
			M.insert(row, col) = value;
		}
	}
	M.makeCompressed();

	Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_vars + b.size());
	for (int i = n_vars; i < rhs.size(); i++)
		rhs[i] = b[static_cast<int64_t>(i) - n_vars];

	Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
	Eigen::VectorXd x_lambda;
	try {
		solver.analyzePattern(M);
		solver.factorize(M);
		if (solver.info() != Eigen::Success) {// Decomposition failed, processing error
			isfail = true;
			return;
		}
		x_lambda = solver.solve(rhs);
	}
	catch (const std::exception& e) {
		isfail = true;
		return;
	}

	x.resize(n_vars);
	x = x_lambda.head(n_vars);
	if (x_lambda.hasNaN()) {
		isfail = true;
		return;
	}
	int e = ((M * x_lambda - rhs).array().abs()).maxCoeff();
	if (e > 0.000001) {
		isfail = true;
		return;
	}
}


void OrbifoldTutteEmbedding::flatten(bool tri_or_square, Eigen::VectorXi all_binds, Eigen::VectorXi p, Eigen::MatrixX3d V,
	Eigen::MatrixX3i T, Eigen::MatrixX2d& V_flat) {
	PosConstraints cons;
	if (tri_or_square) {
		Eigen::Vector2d rhs1 = { 0,0 };
		Eigen::Vector2d rhs2 = { 0,1 };
		Eigen::Vector2d rhs3 = { 1,0 };

		cons.addConstraint(all_binds(p[0]), 1, rhs1);
		cons.addConstraint(all_binds(p[1]), 1, rhs2);
		cons.addConstraint(all_binds(p[2]), 1, rhs3);

		Eigen::Vector2d n1 = { 1,0 };
		Eigen::Vector2d n2 = { 1,1 };
		Eigen::Vector2d n3 = { 0,1 };
		for (int i = p[0] + 1; i <= p[1] - 1; i++) 
			cons.addLineConstraint(all_binds[i], n1, 0);//x=0

		for (int i = p[1] + 1; i <= p[2] - 1; i++) 
			cons.addLineConstraint(all_binds[i], n2.normalized(), sqrt(2) / 2);//x+y=1
		
		for (int i = p[2] + 1; i <= all_binds.size() - 1; i++) 
			cons.addLineConstraint(all_binds[i], n3, 0);//y=0
	}
	else {
		Eigen::Vector2d rhs1 = { -1,1 };
		Eigen::Vector2d rhs2 = { 1,1 };
		Eigen::Vector2d rhs3 = { 1,-1 };
		Eigen::Vector2d rhs4 = { -1,-1 };

		cons.addConstraint(all_binds(p[0]), 1, rhs1);
		cons.addConstraint(all_binds(p[1]), 1, rhs2);
		cons.addConstraint(all_binds(p[2]), 1, rhs3);
		cons.addConstraint(all_binds(p[3]), 1, rhs4);

		Eigen::Vector2d n1 = { 0,1 };
		Eigen::Vector2d n2 = { 1,0 };
		Eigen::Vector2d n3 = { 0,-1 };
		Eigen::Vector2d n4 = { -1,0 };

		for (int i = p[0] + 1; i <= p[1] - 1; i++) {
			cons.addLineConstraint(all_binds[i], n1, 1);//y=1
		}
		for (int i = p[1] + 1; i <= p[2] - 1; i++) {
			cons.addLineConstraint(all_binds[i], n2, 1);//x=1
		}
		for (int i = p[2] + 1; i <= p[3] - 1; i++) {
			cons.addLineConstraint(all_binds[i], n3, 1);//y=-1
		}
		for (int i = p[3] + 1; i <= all_binds.size() - 1; i++) {
			cons.addLineConstraint(all_binds[i], n4, 1);//x=-1
		}
	}

	Eigen::SparseMatrix<double> L(V.rows(), V.rows());
	//L = mean_value_laplacian(V, T);
	L = cotmatrix(V, T);

	//duplicating laplacian to work on each coordinate.
	Eigen::SparseMatrix<double> RealL(L.rows() * 2, L.cols() * 2);
	for (int i = 0; i < L.outerSize(); i++) {
		for (Eigen::SparseMatrix<double> ::InnerIterator it(L, i); it; ++it) {
			int row = it.row();
			int col = it.col();
			double value = it.value();
			RealL.insert(2 * static_cast<int64_t>(row), 2 * static_cast<int64_t>(col)) = value;
			RealL.insert(2 * static_cast<int64_t>(row) + 1, 2 * static_cast<int64_t>(col) + 1) = value;
		}
	}
	RealL.makeCompressed();

	//compute the flattening by solving the boundary conditions while satisfying the convex combination property with L
	Eigen::VectorXd x;
	computeFlattening(cons.A, cons.b, RealL, x);
	if (isfail)
		return;
	Eigen::VectorXd X(x.size() / 2);
	Eigen::VectorXd Y(x.size() / 2);

	for (int i = 0; i < x.size() / 2; i++) {
		X[i] = x[2 * static_cast<int64_t>(i)];
		Y[i] = x[2 * static_cast<int64_t>(i) + 1];
	}
	V_flat << X, Y;
}


void OrbifoldTutteEmbedding::flatten_disk(Eigen::MatrixX3d V, Eigen::MatrixX3i T, Eigen::VectorXi inds, Eigen::MatrixX2d& V_flat,
	std::vector<Eigen::VectorXi>& segs, Eigen::VectorXi b) {
	bool tri_or_square = true;
	if (inds.size() == 3)
		tri_or_square = true;
	else if (inds.size() == 4)
		tri_or_square = false;
	else {
		QMessageBox::warning(nullptr, "cones number error", "disk orbifolds need exactly 3 or 4 cones");
		return;
	}
	flatten(tri_or_square, b, inds, V, T, V_flat);//flatten the disk mesh to one of the orbifolds. 
	if (isfail)
		return;
	// caculate the boundary_segments
	for (int i = 0; i < inds.size(); i++) {
		if (i < inds.size() - 1) {
			Eigen::VectorXi seg(inds[static_cast<int64_t>(i) + 1] - inds[i] + 1);
			for (int j = inds[i]; j <= inds[static_cast<int64_t>(i)+1]; j++) 
				seg[static_cast<int64_t>(j) - inds[i]] = b[j];
			segs.emplace_back(seg);
		}
		else {
			Eigen::VectorXi seg(b.size() - inds[i] + inds[0] + 1);
			for (int j = inds[i]; j < b.size(); j++) 
				seg[static_cast<int64_t>(j) - inds[i]] = b[j];
			for (int j = 0; j <= inds[0]; j++) 
				seg[seg.size() - inds[0] + j - 1] = b[j];
			segs.emplace_back(seg);
		}
	}
}


void OrbifoldTutteEmbedding::compute_map_from_disk_embeddings(Eigen::MatrixX2d V_source, Eigen::MatrixX3i T_source,
	Eigen::MatrixX2d V_target, Eigen::MatrixX3i T_target, std::vector<Eigen::VectorXi> boundary_source, 
	std::vector<Eigen::VectorXi> boundary_target, Eigen::SparseMatrix<double>& BC) {
	Eigen::VectorXi tri_ind(V_source.rows());
	Eigen::MatrixX3d bc = Eigen::MatrixX3d::Ones(V_source.rows(), 3) / 3;

	//finds the barycentric coorinates of all interior vertices;
	for (int i = 0; i < V_source.rows(); i++) {// Traverse all source points
		Point_2d p;
		p.x = V_source(i, 0);
		p.y = V_source(i, 1);
		for (int j = 0; j < T_target.rows(); j++) {//Traverse all triangles
			std::vector<Point_2d> vertices;
			for (int k = 0; k < 3; k++) {
				Point_2d temp_p;
				temp_p.x = V_target(T_target(j, k), 0);
				temp_p.y = V_target(T_target(j, k), 1);
				vertices.emplace_back(temp_p);
			}
			if (isPointInConvexPolygon(vertices, p)) {
				double alpha = 0.0, beta = 0.0, gamma = 0.0;
				cal_barycentric(vertices[0], vertices[1], vertices[2], p, alpha, beta, gamma);
				bc(i, 0) = alpha;
				bc(i, 1) = beta;
				bc(i, 2) = gamma;
				tri_ind(i) = j;
				break;
			}
			//if (j == (T_target.rows() - 1)) {//Unable to find the inner triangle, turn to find the nearest triangle.
			//	int nearestTriangle = -1;
			//	double minDistance = std::numeric_limits<double>::max();
			//	Eigen::Vector2d point;
			//	point[0] = p.x;
			//	point[1] = p.y;
			//	Eigen::Vector2d centroid;
			//	for (int k = 0; k < T_target.rows(); ++k) {
			//		Eigen::Vector2d v1 = V_target.row(T_target(k, 0));
			//		Eigen::Vector2d v2 = V_target.row(T_target(k, 1));
			//		Eigen::Vector2d v3 = V_target.row(T_target(k, 2));
			//		centroid = (v1 + v2 + v3) / 3.0;
			//		double distance = (centroid - point).norm();
			//		if (distance < minDistance) {
			//			minDistance = distance;
			//			nearestTriangle = k;
			//		}
			//	}
			//	tri_ind[i] = nearestTriangle;
			//	vertices.clear();
			//	for (int k = 0; k < 3; k++) {
			//		Point_2d temp_p;
			//		temp_p.x = V_target(T_target(nearestTriangle, k), 0);
			//		temp_p.y = V_target(T_target(nearestTriangle, k), 1);
			//		vertices.emplace_back(temp_p);
			//	}
			//	double alpha = 0.0, beta = 0.0, gamma = 0.0;
			//	cal_barycentric(vertices[0], vertices[1], vertices[2], p, alpha, beta, gamma);
			//	bc(i, 0) = alpha;
			//	bc(i, 1) = beta;
			//	bc(i, 2) = gamma;
			//}
		}
	}

	//go over each boundary segment (an "edge" of the disk orbifold polygon)
	for (int i = 0; i < boundary_source.size(); i++) {
		//current boundary segment
		Eigen::VectorXi p_target = boundary_target[i];
		Eigen::VectorXi p_source = boundary_source[i];

		// first handling the cones - which necessarily lie as end and start vertices of the boundary segment
		Eigen::Vector2i startend_source, startend_target;
		startend_source << p_source[0],
			p_source[p_source.size() - 1];
		startend_target << p_target[0],
			p_target[p_target.size() - 1];

		//find any triangle containing the boundary vertices
		for (int j = 0; j < 2; j++) {
			int k; // used to record the number of triangular surface
			int idx = 0; //used to record the index number on a triangular surface
			for (k = 0; k < T_target.rows(); k++) {
				if (T_target(k, 0) == startend_target[j]) {
					idx = 0;
					break;
				}
				if (T_target(k, 1) == startend_target[j]) {
					idx = 1;
					break;
				}
				if (T_target(k, 2) == startend_target[j]) {
					idx = 2;
					break;
				}
			}
			int Tind = k;
			tri_ind[startend_source[j]] = Tind;
			bc.row(startend_source[j]) = Eigen::Vector3d::Zero();
			bc(startend_source[j], idx) = 1;
		}

		// mapping boundary vertices which are not cones
		// take the vector of the direction of the "infinite line" on which the edge of 
		// the orbifold polygon lies(line connecting the start and end cones)
		Eigen::Vector2d vv = V_target.row(p_target[p_target.size() - 1]) - V_target.row(p_target[0]);

		// do not wish to compute the map for the two cones so remove them
		p_source.segment(0, p_source.size() - 2) = p_source.segment(1, p_source.size() - 2);
		p_source.conservativeResize(p_source.size() - 2);

		// take the position along the infinite line vv of all boundary vertices both on source and target
		Eigen::VectorXd line_distance_target(p_target.size()), line_distance_source(p_source.size());
		for (int j = 0; j < p_target.size(); j++)
			line_distance_target[j] = V_target.row(p_target[j]).dot(vv);

		for (int j = 0; j < p_source.size(); j++)
			line_distance_source[j] = V_source.row(p_source[j]).dot(vv);

		//map each source vertex
		for (int j = 0; j < line_distance_source.size(); j++) {
			// find the two consecutive boundary vertices on target between which source vertex lies(all projected onto the infinite line vv)
			std::vector<int> ind_of_containing_segment;
			for (int k = 0; k < line_distance_target.size() - 1; k++) 
				if (line_distance_target[k]<line_distance_source[j] && 
					line_distance_target[static_cast<int64_t>(k) + 1]>line_distance_source[j]) 
					ind_of_containing_segment.emplace_back(k);
			
			if (ind_of_containing_segment.size() != 1) {
				isfail = true;
				return;
			}

			// Compute the barycentric coordinates of source vertex w.r.t the two found target vertices.
			// barycentric coords of x in [a, b] is (x - a) / (b - a)
			double c = (line_distance_source[j] - line_distance_target[ind_of_containing_segment[0]]) / 
				(line_distance_target[static_cast<int64_t>(ind_of_containing_segment[0]) + 1] - 
					line_distance_target[ind_of_containing_segment[0]]);
			//now find the triangle which has the containing boundary segment as its edge
			std::vector<int> triind;
			int k;
			for (k = 0; k < T_target.rows(); k++) {
				int flag1 = 0, flag2 = 0, flag3 = 0;
				if (T_target(k, 0) == p_target[ind_of_containing_segment[0]] || T_target(k, 0) == 
					p_target[static_cast<int64_t>(ind_of_containing_segment[0]) + 1])
					flag1 = 1;
				if (T_target(k, 1) == p_target[ind_of_containing_segment[0]] || T_target(k, 1) == 
					p_target[static_cast<int64_t>(ind_of_containing_segment[0]) + 1])
					flag2 = 1;
				if (T_target(k, 2) == p_target[ind_of_containing_segment[0]] || T_target(k, 2) == 
					p_target[static_cast<int64_t>(ind_of_containing_segment[0]) + 1])
					flag3 = 1;
				if (flag1 + flag2 + flag3 == 2)
					triind.emplace_back(k);
			}
			if (triind.size() != 1) {
				isfail = true;
				return;
			}
			//inserting the correct triangle and barycentric coordinates into the list
			tri_ind(p_source[j]) = triind[0];
			//take the vertex indices of the triangle and insert the barycentric coordinates in the correct order according to them
			Eigen::Vector3i TT = T_target.row(triind[0]);
			//first zero-out all of them
			bc.row(p_source[j]) = Eigen::VectorXd::Zero(3);
			//since vertex lies on the edge, one coordinate is left as zero, other two are filled as c and 1 - c
			int TT_idx1, TT_idx2;
			for (int l = 0; l < 3; l++) {
				if (TT[l] == p_target[ind_of_containing_segment[0]])
					TT_idx1 = l;
				if (TT[l] == p_target[static_cast<int64_t>(ind_of_containing_segment[0]) + 1])
					TT_idx2 = l;
			}
			bc(p_source[j], TT_idx1) = 1 - c;
			bc(p_source[j], TT_idx2) = c;
		}
	}
	if (tri_ind.hasNaN() || tri_ind.minCoeff() < 0||tri_ind.maxCoeff()>=T_target.rows()) {
		isfail = true;
		return;
	}
	for (int i = 0; i < bc.rows(); i++) {
		if (std::abs(bc.row(i).sum() - 1) > 0.0001) {

		}
	}
	//make sure all source vertices were matched these 3 lines are for making sure matrix has right size horizontal dimension
	Eigen::MatrixX3d V(V_source.rows() + 1, 3);
	Eigen::MatrixX3i J(V_source.rows() + 1, 3), I(V_source.rows() + 1, 3);
	for (int i = 0; i < V_source.rows(); i++) 
		J.row(i) = T_target.row(tri_ind[i]);
	J.row(V_source.rows()) = Eigen::Vector3i::Ones() * (V_target.rows() - 1);

	Eigen::VectorXi I_1(V_source.rows() + 1);
	for (int i = 0; i < V_source.rows(); i++) 
		I_1[i] = i;
	
	I_1[I_1.size() - 1] = 0;
	I << I_1, I_1, I_1;
	V << bc,
		Eigen::MatrixX3d::Zero(1, 3);
	for (int i = 0; i <= V_source.rows(); i++) {
		BC.coeffRef(I(i, 0), J(i, 0)) = BC.coeff(I(i, 0), J(i, 0)) + V(i, 0);
		BC.coeffRef(I(i, 1), J(i, 1)) = BC.coeff(I(i, 1), J(i, 1)) + V(i, 1);
		BC.coeffRef(I(i, 2), J(i, 2)) = BC.coeff(I(i, 2), J(i, 2)) + V(i, 2);
	}
	BC.makeCompressed();
}


void OrbifoldTutteEmbedding::map_disks(Eigen::MatrixX3d V1, Eigen::MatrixX3i T1, Eigen::VectorXi sc1, Eigen::MatrixX3d V2,
	Eigen::MatrixX3i T2, Eigen::VectorXi sc2, Eigen::VectorXi b1, Eigen::VectorXi b2, 
	Eigen::SparseMatrix<double>& BC_1to2) {
	Eigen::MatrixX2d V_flat1(V1.rows(), 2), V_flat2(V2.rows(), 2);
	std::vector<Eigen::VectorXi> segs1, segs2;
	flatten_disk(V1, T1, sc1, V_flat1, segs1, b1);
	if (isfail)
		return;
	flatten_disk(V2, T2, sc2, V_flat2, segs2, b2);
	if (isfail)
		return;
	compute_map_from_disk_embeddings(V_flat1, T1, V_flat2, T2, segs1, segs2, BC_1to2);
	if (isfail)
		return;
}


bool OrbifoldTutteEmbedding::disk_map(Eigen::MatrixX3d V1, Eigen::MatrixX3i T1, Eigen::VectorXi b1,
	Eigen::MatrixX3d V2, Eigen::MatrixX3i T2, Eigen::VectorXi b2, Eigen::VectorXi sc1, 
	Eigen::VectorXi sc2, Eigen::MatrixX3d& verticesMapped) {
	isfail = false;

	Eigen::SparseMatrix<double> BC_1to2(V1.rows(), V2.rows());
	map_disks(V1, T1, sc1, V2, T2, sc2, b1, b2, BC_1to2);
	if (isfail) {
		return false;
	}
	if (BC_1to2.rows() == 0) {
		isfail = true;
		return false;
	}
	
	verticesMapped = BC_1to2 * V2;
	return true;
}