/*
 * LShapeAnalyserFunction.cpp
 *
 *  Created on: August 30, 2023
 *      Author: Jialong Zhang
 */

#include <filesystem>

#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/eigen.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdTree.h>

#include "LShapeAnalyserFunction.h"

#define PI acos(-1)

void DeleteDiretoryContents(const std::string& diretory) {//Recursively delete folder
    for (const auto& entry : std::filesystem::directory_iterator(diretory)) {
        const auto& path = entry.path();
        if (std::filesystem::is_directory(path)) {
            DeleteDiretoryContents(path.string());
            std::filesystem::remove(path);
        }
        else {
            std::filesystem::remove(path);
        }
    }
}


int Dsearchn(pcl::PointCloud<pcl::PointXYZ>::Ptr pc, Eigen::VectorXi boundary, pcl::PointXYZ verticesKeyPointI) {
	pcl::PointXYZ searchPoint = verticesKeyPointI;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr boundaryPc(new pcl::PointCloud<pcl::PointXYZ>);
	boundaryPc->resize(boundary.size());
	for (int i = 0; i < boundary.size(); i++) {
		boundaryPc->points[i] = pc->points[boundary[i]];
	}
	kdTree->setInputCloud(boundaryPc);
	std::vector<int> pointIdxNKNSearch(1);
	std::vector<float> pointNKNSquaredDistance(1);
	kdTree->nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
	return pointIdxNKNSearch[0];
}


double EvaluateScore(Eigen::MatrixXd V1, Eigen::MatrixXd V2, Eigen::MatrixX3i T1) {
	Eigen::MatrixXd sub(V1.rows(), V1.cols());
	sub = V1 - V2;
	Eigen::VectorXd distance(V1.rows());
	for (int i = 0; i < sub.rows(); i++) {
		if (sub.row(i).sum() < 0)
			distance[i] = -sub.row(i).dot(sub.row(i));
		else
			distance[i] = sub.row(i).dot(sub.row(i));
	}
	double score;
	score = distance.mean();

	return score;
}


Eigen::MatrixX3d GenerateAvg(std::vector<Eigen::MatrixX3d> V, int fnum, int verticesNum) {
	Eigen::MatrixX3d verticesAvg = Eigen::MatrixX3d::Zero(V[0].rows(), 3);
	for (int i = 0; i < fnum; i++) {
		verticesAvg += V[i];
	}
	verticesAvg /= fnum;
	return verticesAvg;
}


// Calculate the cross product of vectors
double crossProduct(const Point_2d a, const Point_2d b, const Point_2d c) {
    double vx1 = b.x - a.x;
    double vy1 = b.y - a.y;
    double vx2 = c.x - b.x;
    double vy2 = c.y - b.y;

    return vx1 * vy2 - vx2 * vy1;
}

// Determine whether point p is to the left of line segment ab
bool isLeft(const Point_2d& a, const Point_2d& b, const Point_2d& p) {
    return ((b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y)) > 0;
}

// Judge whether the polygon is Concave polygon
bool isConcavePolygon(const std::vector<Point_2d>& polygon) {
    int n = polygon.size();
    for (int i = 0; i < n; ++i) {
        double result = crossProduct(polygon[i], polygon[(i + 1) % n], polygon[(i + 2) % n]);
        if (result < 0) {
            return true; // If the cross product of adjacent line segments is negative, it is judged to be a Concave polygon
        }
    }

    return false; // The cross product of all adjacent line segments is non negative and is a Convex polygon
}

// Judge whether point p is in the Convex polygon formed by points set vertices
bool isPointInConvexPolygon(const std::vector<Point_2d>& vertices, const Point_2d& p) {
    // Get the number of vertices of Convex polygon
    int n = vertices.size();

    // Take a vertex of a Convex polygon (for example, the first vertex)
    Point_2d extreme = { 100000, p.y };  // Assuming a point far enough away

    int count = 0, i = 0;
    do {
        int next = (i + 1) % n;

        // Determine whether points p and extreme are located on both sides of the edge vertices [i] - vertices [next]
        if (((vertices[i].y > p.y) != (vertices[next].y > p.y)) &&
            (p.x < (static_cast<double>(vertices[next].x) - vertices[i].x) * (static_cast<double>(p.y) - vertices[i].y) / (static_cast<double>(vertices[next].y) - vertices[i].y) + vertices[i].x))
            count++;

        i = next;
    } while (i != 0);

    // Odd numbers on the inside, even numbers on the outside
    return (count % 2 == 1);
}

// Judge whether point p is in the Concave polygon
bool isPointInConcavePolygon(const std::vector<Point_2d>& polygon, const Point_2d& p) {
    int count = 0;
    int n = polygon.size();

    // Traverse the edges of a polygon
    for (int i = 0; i < n; ++i) {
        const Point_2d& p1 = polygon[i];
        const Point_2d& p2 = polygon[(i + 1) % n];

        // Determine whether point p is to the left of edge p1p2
        if (isLeft(p1, p2, p)) {
            ++count;
        }
    }

    // If point p is within the polygon, then the intersection points between the ray and the polygon edge are even, otherwise they are odd
    return (count % 2 == 1);
}


void Procrustes(Eigen::MatrixX3d referencePoints, Eigen::MatrixX3d& targetPoints) {
    Eigen::Vector3d referenceCentroid = referencePoints.colwise().mean();
    Eigen::Vector3d targetCentroid = targetPoints.colwise().mean();

    Eigen::MatrixX3d refCentered = referencePoints.rowwise() - referenceCentroid.transpose();
    Eigen::MatrixX3d tarCentered = targetPoints.rowwise() - targetCentroid.transpose();

    Eigen::Matrix3d covariance = refCentered.adjoint() * tarCentered;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d rotationMatrix = svd.matrixU() * svd.matrixV().transpose();

    Eigen::Vector3d t = targetCentroid - rotationMatrix * referenceCentroid;

    Eigen::MatrixX3d tran = targetPoints * rotationMatrix.transpose() + t.transpose().replicate(targetPoints.rows(), 1);
    targetPoints = tran;
}


void RemoveVerticesAndFacesByValue(pcl::PolygonMesh::Ptr& mesh, const std::vector<int> indices, std::vector<pcl::Vertices> facesOriginal)
{
    // Create a new mesh
    pcl::PolygonMesh::Ptr newmesh(new pcl::PolygonMesh);

    // create the map index and add the polygon in newmesh
    std::vector<int> Map;
    for (int i = 0; i < facesOriginal.size(); i++) {//Traverse every polygon face
        pcl::Vertices polygon = facesOriginal[i];
        for (int j = 0; j < polygon.vertices.size(); j++) {
            if (std::find(indices.begin(), indices.end(), polygon.vertices[j]) == indices.end())//not found
                break;//skip this polygon
            if (j == polygon.vertices.size() - 1) {
                newmesh->polygons.push_back(mesh->polygons[i]);
                for (int k = 0; k < polygon.vertices.size(); k++) {
                    if (std::find(Map.begin(), Map.end(), polygon.vertices[k]) == Map.end())
                        Map.emplace_back(polygon.vertices[k]);
                }
            }
        }
    }
    std::sort(Map.begin(), Map.end());

    // extract the point within indices
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::PointCloud<pcl::PointXYZ>* cloud = new pcl::PointCloud<pcl::PointXYZ>;
    pcl::fromPCLPointCloud2(mesh->cloud, *cloud);
    extract.setInputCloud(cloud->makeShared());
    pcl::PointIndices::Ptr pointIndices(new pcl::PointIndices);
    pointIndices->indices = Map;
    extract.setIndices(pointIndices);
    extract.setNegative(false); //set false to remain the indices point
    extract.filter(*cloud);
    pcl::toPCLPointCloud2(*cloud, newmesh->cloud);//set the points

    //Reorder index of the faces
    for (int i = 0; i < newmesh->polygons.size(); i++) {
        pcl::Vertices& NV = newmesh->polygons[i];
        for (int j = 0; j < NV.vertices.size(); j++) {
            auto it = std::find(Map.begin(), Map.end(), NV.vertices[j]);
            int idx = std::distance(Map.begin(), it);
            NV.vertices[j] = idx;
        }
    }
    *mesh = *newmesh;
}


Eigen::MatrixXd RotatePoint(Eigen::MatrixXd V, float theta, Eigen::Vector3d rotateAxis) {
    Eigen::VectorXd x = V.col(0).array();
    Eigen::VectorXd y = V.col(1).array();
    Eigen::VectorXd z = V.col(2).array();

    double vx = rotateAxis[0];
    double vy = rotateAxis[1];
    double vz = rotateAxis[2];

    double r = theta * PI / 180;
    double sinr = std::sin(r);
    double cosr = std::cos(r);

    Eigen::VectorXd rotateX, rotateY, rotateZ;
    for (int i = 0; i < x.size(); i++) {
        rotateX[i] = (vx * vx * (1 - cosr) + cosr) * x[i] + (vx * vy * (1 - cosr) - vz * sinr) * y[i] + (vx * vz * (1 - cosr) + vy * sinr) * z[i];
        rotateY[i] = (vx * vy * (1 - cosr) + vz * sinr) * x[i] + (vy * vy * (1 - cosr) + cosr) * y[i] + (vy * vz * (1 - cosr) - vx * sinr) * z[i];
        rotateZ[i] = (vx * vz * (1 - cosr) - vy * sinr) * x[i] + (vy * vz * (1 - cosr) + vx * sinr) * y[i] + (vz * vz * (1 - cosr) + cosr) * z[i];
    }

    Eigen::MatrixXd verticesRotated;
    verticesRotated << rotateX, rotateY, rotateZ;
    return verticesRotated;
}


Eigen::Vector4d SolveSurface(pcl::PointXYZ V1, pcl::PointXYZ V2, double a1, double b1, double c1) {
    double EPS = 0.00001;
    if ((std::abs(V1.y - V2.y) < EPS) && (std::abs(V1.z - V2.z) < EPS)) {
        double a = 0.0;
        double b = 1.0;
        Eigen::MatrixXd A(2, 2);
        A(0, 0) = V1.z;
        A(0, 1) = 1;
        A(1, 0) = c1;
        A(1, 1) = 0;

        Eigen::MatrixXd B(2, 1);
        B(0, 0) = -V1.y;
        B(1, 0) = -b1;
        Eigen::MatrixXd X;
        X = A.colPivHouseholderQr().solve(B);
        double c = X(0, 0);
        double d = X(1, 0);
        Eigen::Vector4d ABCD(a, b, c, d);
        return ABCD;
    }
    else {
        double a = 1.0;
        Eigen::MatrixXd A(3, 3);
        A(0, 0) = V1.y;
        A(0, 1) = V1.z;
        A(0, 2) = 1;
        A(1, 0) = V2.y;
        A(1, 1) = V2.z;
        A(1, 2) = 1;
        A(2, 0) = b1;
        A(2, 1) = c1;
        A(2, 2) = 0;

        Eigen::MatrixXd B(3, 1);
        B(0, 0) = -a * V1.x;
        B(1, 0) = -a * V2.x;
        B(2, 0) = -a * a1;

        Eigen::MatrixXd X;
        X = A.colPivHouseholderQr().solve(B);
        double b = X(0, 0);
        double c = X(1, 0);
        double d = X(2, 0);
        Eigen::Vector4d ABCD(a, b, c, d);
        return ABCD;
    }
}


std::vector<std::string> SplitString(const std::string& input, const std::string& delimiters) {
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string token;

    while (std::getline(ss, token)) {
        size_t start = 0;
        size_t end = 0;

        while ((end = token.find_first_of(delimiters, start)) != std::string::npos) {
            if (end != start) {
                result.push_back(token.substr(start, end - start));
            }
            start = end + 1;
        }

        if (start < token.length()) {//Fill in the last character of a line
            result.push_back(token.substr(start));
        }
    }

    return result;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr TranslateAndNormalizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // Step 1: Calculate centroid
    Eigen::Vector4d centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    // Step 2: Calculate translation vector
    Eigen::Affine3d translation(Eigen::Translation3d(-centroid[0], -centroid[1], -centroid[2]));

    // Step 3: Perform translation operation
    pcl::PointCloud<pcl::PointXYZ>::Ptr transCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *transCloud, translation);

    // Step 4: Calculate the sum of Euclidean distances
    double sumSquaredDistances = 0;
    for (const auto& point : transCloud->points)
        sumSquaredDistances += pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2);

    // Step 5: Calculate standardized scaling factor
    double scaleFactor = 1.0 / sqrt(sumSquaredDistances);

    // Step 6: Perform scaling operations
    for (int i = 0; i < transCloud->size(); ++i) {
        pcl::PointXYZ& point = transCloud->at(i);
        point.getVector3fMap() *= scaleFactor;
    }
    return transCloud;
}