/*
 * LShapeAnalyserFunction.h
 *
 *  Created on: August 30, 2023
 *      Author: Jialong Zhang
 */

#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

struct Point_2d
{
	double x, y;
};

void DeleteDiretoryContents(const std::string& diretory);

int Dsearchn(pcl::PointCloud<pcl::PointXYZ>::Ptr pc, Eigen::VectorXi boundary, pcl::PointXYZ verticesKeyPointI);

double EvaluateScore(Eigen::MatrixXd V1, Eigen::MatrixXd V2, Eigen::MatrixX3i T1);

Eigen::MatrixX3d GenerateAvg(std::vector<Eigen::MatrixX3d> V, int fnum, int verticesNum);

double crossProduct(const Point_2d a, const Point_2d b, const Point_2d c);

bool isLeft(const Point_2d& a, const Point_2d& b, const Point_2d& p);

bool isConcavePolygon(const std::vector<Point_2d>& polygon);

bool isPointInConvexPolygon(const std::vector<Point_2d>& vertices, const Point_2d& p);

bool isPointInConcavePolygon(const std::vector<Point_2d>& polygon, const Point_2d& p);

void Procrustes(Eigen::MatrixX3d referencePoints, Eigen::MatrixX3d& targetPoints);

void RemoveVerticesAndFacesByValue(pcl::PolygonMesh::Ptr& mesh, const std::vector<int> indices, std::vector<pcl::Vertices> facesOriginal);

Eigen::MatrixXd RotatePoint(Eigen::MatrixXd V, float theta, Eigen::Vector3d rotateAxis);

Eigen::Vector4d SolveSurface(pcl::PointXYZ V1, pcl::PointXYZ V2, double a1, double b1, double c1);

std::vector<std::string> SplitString(const std::string& input, const std::string& delimiters);

pcl::PointCloud<pcl::PointXYZ>::Ptr TranslateAndNormalizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);