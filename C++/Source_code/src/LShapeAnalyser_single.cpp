/*
 * LShapeAnalyser_single.cpp
 *
 *  Created on: August 30, 2023
 *      Author: Jialong Zhang
 */

#include <direct.h>
#include <random>
#include <io.h>
#include <sstream>
#include <chrono>
#include <filesystem>

#include <QMessageBox>
#include <QFileDialog>
#include <QTextEdit>
#include <QProgressDialog>
#include <QProgressBar>
#include <QTableWidget>
#include <QFile>
#include <QTextStream>
#include <QDir>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <pcl/common/transforms.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/poisson.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

#include "LShapeAnalyser_single.h"
#include "LShapeAnalyserFunction.h"
#include "OrbifoldTutteEmbedding.h"

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
typedef Surface_mesh::Vertex_index Vertex_index;
typedef Surface_mesh::Halfedge_index Halfedge_index;


void LShapeAnalyser_single::setProjectName(const QString name) {
	_projectName = name;
}


void LShapeAnalyser_single::setDLCPath() {
	QString currentPath = QDir::currentPath();
	QDir dir(currentPath + "/" + _projectName);
	if (dir.exists("DLC")) 
		ui.DLCPathLineEdit->setText(currentPath + "/" + _projectName + "/DLC");
}


LShapeAnalyser_single::LShapeAnalyser_single(QWidget* parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	//Signal Connection
	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(exitSingle()));
	connect(ui.actionImport, SIGNAL(triggered()), this, SLOT(importDepth()));
	connect(ui.openDLCButton, SIGNAL(clicked()), this, SLOT(openDLC()));
	connect(ui.startConvertButton, SIGNAL(clicked()), this, SLOT(startConvert()));
	connect(ui.cancelConvertButton, SIGNAL(clicked()), this, SLOT(cancel()));
	connect(ui.loadTruthButton, SIGNAL(clicked()), this, SLOT(loadTruth()));
	connect(ui.resetTruthButton, SIGNAL(clicked()), this, SLOT(resetTruth()));
	connect(ui.importTruthButton, SIGNAL(clicked()), this, SLOT(importTruth()));
	connect(ui.saveTruthButton, SIGNAL(clicked()), this, SLOT(saveTruth()));
	connect(ui.startEvaluateButton, SIGNAL(clicked()), this, SLOT(startEvaluate()));
	connect(ui.cancelEvaluateButton, SIGNAL(clicked()), this, SLOT(cancel()));
	connect(ui.trainButton, SIGNAL(clicked()), this, SLOT(train()));
	connect(ui.browseDLCButton, SIGNAL(clicked()), this, SLOT(browseDLC()));
	connect(ui.editParaButton, SIGNAL(clicked()), this, SLOT(editPara()));
	connect(ui.scoreButton, SIGNAL(clicked()), this, SLOT(score()));
	connect(ui.exportScoreButton, SIGNAL(clicked()),this,SLOT(exportScore()));
}


void LShapeAnalyser_single::cancel() {
	_stopExecution = true;
}


void LShapeAnalyser_single::exitSingle() {
	QMessageBox::StandardButton reply;
	reply = QMessageBox::question(nullptr, "Exit", "Confirm to exit", QMessageBox::Yes | QMessageBox::No);
	if (reply == QMessageBox::Yes) {
		this->close();
	}
	else {
		return;
	}
}


void LShapeAnalyser_single::importDepth() {
	QString folderPath = QFileDialog::getExistingDirectory(this, "Select depth image folder", "", QFileDialog::ShowDirsOnly);
	if (!folderPath.isEmpty()) {
		char mainPath[_MAX_PATH];
		if (!_getcwd(mainPath, _MAX_PATH)) {//Get the main path
			QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
			return;
		}

		std::string depthPath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\depth";
		std::string grayPath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\gray";

		if (_access(depthPath.c_str(), 0) == 0) { //Determine if the depth folder exists
			DeleteDiretoryContents(depthPath); //Delete old
			std::filesystem::remove(depthPath);
		}
		if (_mkdir(depthPath.c_str()) != 0) { //Recreate the depth folder 
			QMessageBox::warning(nullptr, "Folder depth create failure", "Folder depth create failure");
			return;
		}

		if (_access(grayPath.c_str(), 0) == 0) { //Determine if the gray folder exists
			DeleteDiretoryContents(grayPath);
			std::filesystem::remove(grayPath);
		}
		if (_mkdir(grayPath.c_str()) != 0) { //Recreate the gray folder
			QMessageBox::warning(nullptr, "Folder gray create failure", "Folder gray create failure");
			return;
		}

		// Calculate the number of files
		int numFiles = 0;
		for (const auto& entry : std::filesystem::directory_iterator(folderPath.toStdString())) {
			if (std::filesystem::is_regular_file(entry)) {
				numFiles++;
			}
		}
		int quantile = floor(log10(numFiles) + 1); // Quantile of document

		// Copy depth image file from source path to create new folder and convert them to gray image
		_finddatai64_t fileInfo;
		intptr_t fileHandel = 0;
		if ((fileHandel = _findfirst64((folderPath.toStdString() + "\\*.png").c_str(), &fileInfo)) != -1) {
			int number = 0;
			cv::VideoWriter writer;
			QProgressDialog progressDiglog;
			progressDiglog.setLabelText("Importing...");
			progressDiglog.setRange(0, 100);
			Qt::WindowFlags flags = progressDiglog.windowFlags();
			progressDiglog.setWindowFlags(flags | Qt::WindowStaysOnTopHint);
			do {
				progressDiglog.setValue(number * 100 / numFiles);
				QApplication::processEvents(); //Update UI response
				number++;
				std::string sourceFile = folderPath.toStdString() + "\\" + fileInfo.name;
				std::string depthTargetFile;
				std::stringstream idx;
				idx << std::setw(quantile) << std::setfill('0') << number; //Fill in '0' before numbers

				depthTargetFile = depthPath + "\\depth_" + idx.str() + ".png";
				try {
					std::filesystem::copy_file(sourceFile, depthTargetFile, std::filesystem::copy_options::overwrite_existing);
				}
				catch (const std::filesystem::filesystem_error& e) {
					QMessageBox::warning(nullptr, "Copy error", "Copy error");
				}

				// Write the gray image and the corresponding video with extension .avi
				cv::Mat depthImage = cv::imread(sourceFile, cv::IMREAD_UNCHANGED);
				if (depthImage.empty()) {
					QMessageBox::warning(nullptr, "Can't read the depth image", "Can't read the depth image");
					return;
				}
				double depthMin, depthMax;
				cv::minMaxLoc(depthImage, &depthMin, &depthMax);
				int h = depthImage.cols;
				int w = depthImage.rows;
				cv::Mat grayImage = cv::Mat::zeros(depthImage.size(), CV_8UC1);
				//Grayscale pixels
				for (int col = 0; col < h; col++) {
					for (int row = 0; row < w; row++) {
						grayImage.at<uchar>(row, col) = (depthImage.at<unsigned short>(row, col) - depthMin) / (depthMax - depthMin) * 255.0;
					}
				}

				std::string grayTargetFile = grayPath + "\\gray_" + idx.str() + ".png";
				cv::imwrite(grayTargetFile, grayImage);
				if (number == 1)
					writer = cv::VideoWriter(grayPath + "\\Synthetic.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 32, cv::Size(h, w), true);

				cv::Mat image = cv::imread(grayTargetFile);
				writer << image;
			} while (_findnext64(fileHandel, &fileInfo) == 0);
			progressDiglog.close();
			writer.release();
			QTextEdit* outputInfo = ui.outputInfoText;
			outputInfo->append("Import depth file from " + folderPath + ", and the gray image and synthetic video have been created.");
			QMessageBox::information(nullptr, "Import successfully", "Import successfully");
		}
		else {
			//fileHandel==-1
			QMessageBox::warning(nullptr, "Not file in this folder", "Not file in this folder");
			return;
		}
	}
	else {
		return;
	}
}


void LShapeAnalyser_single::openDLC() {
	char mainPath[_MAX_PATH];
	if (!_getcwd(mainPath, _MAX_PATH)) {
		QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
		return;
	}

	std::string syn = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\" + "gray\\Synthetic.avi";
	if (_access(syn.c_str(), 0) != 0) {
		QMessageBox::warning(nullptr, "Not video exist", "Not video with extension .avi exist, please import\\reimport file firstly");
		return;
	}

	/* @article{ Lauer2022MultianimalPE,
	 title = {Multi - animal pose estimation, identification and tracking with DeepLabCut},
	 author = {Jessy Lauer and Mu Zhou and Shaokai Ye and William Menegas and
	 Steffen Schneider and Tanmay Nath and Mohammed Mostafizur Rahman and
	 Valentina Di Santo and Daniel Soberanes and Guoping Feng and
	 Venkatesh fileNumber.Murthy and George Lauder and Catherine Dulac and
	 M.Mathis and Alexander Mathis},
	 journal = {Nature Methods},
	 year = {2022},
	 volume = {19},
	 pages = {496 - 504} }*/
	std::string command = "conda activate DEEPLABCUT && python -m deeplabcut";
	system(command.c_str());
}


void LShapeAnalyser_single::startConvert() {
	auto startTime = std::chrono::high_resolution_clock::now();
	char mainPath[_MAX_PATH];
	if (!_getcwd(mainPath, _MAX_PATH)) {
		QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
		return;
	}

	std::filesystem::path markPath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\gray";
	std::filesystem::path depthPath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\depth";
	if (std::filesystem::is_empty(depthPath)) {
		QMessageBox::warning(nullptr, "Depth folder empty", "Depth folder is empty");
		return;
	}
	// Calculate the number of files
	int numFiles = 0;
	for (const auto& entry : std::filesystem::directory_iterator(depthPath))
		if (std::filesystem::is_regular_file(entry) && entry.path().extension() == ".png")
			numFiles++;

	double fx = ui.fxLineEdit->text().toDouble();
	double fy = ui.fyLineEdit->text().toDouble();
	double cx = ui.cxLineEdit->text().toDouble();
	double cy = ui.cyLineEdit->text().toDouble();

	//Extract the predicted marker point coordinates
	_finddatai64_t fileInfo;
	intptr_t fileHandel = 0;
	if ((fileHandel = _findfirst64((markPath.string() + "\\*.csv").c_str(), &fileInfo)) == -1) {
		QMessageBox::warning(nullptr, "No key point infomation", "Key points have not been detected yet.");
		return;
	}

	std::string csvFilePath = markPath.string() + "\\" + fileInfo.name;
	std::string meshPath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\mesh";
	std::string pcPath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\pointcloud";

	//clear the old mesh and pointcloud file
	if (_access(meshPath.c_str(), 0) == 0)
		DeleteDiretoryContents(meshPath);

	if (_access(pcPath.c_str(), 0) == 0)
		DeleteDiretoryContents(pcPath);

	//read keyPoint coordinate from table element 
	std::vector<std::vector<double>> markPoint;
	std::vector<std::vector<double>> likelihood;
	std::string lineStr;
	std::ifstream file(csvFilePath);

	int notPointCount = 0;
	while (std::getline(file, lineStr)) {
		if (notPointCount < 3) {//The first three lines of the file are attribute names
			notPointCount++;
			continue;
		}
		int count = 0;
		std::vector<double> pointRow;
		std::vector<double> likeRow;
		std::stringstream ss(lineStr);

		std::string cell;
		while (std::getline(ss, cell, ',')) {
			if (count != 0) {// the first col is number
				double value = std::stod(cell);
				if (count % 3 == 0) {
					likeRow.emplace_back(value);
				}
				else {
					pointRow.emplace_back(value);
				}
			}
			count++;
		}
		markPoint.emplace_back(pointRow);
		likelihood.emplace_back(likeRow);
	}
	file.close();
	int markNum = markPoint[0].size() / 2;

	QTextEdit* outputInfo = ui.outputInfoText;
	outputInfo->append("Start convert to mesh.");

	// Process each image
	int canNotMapCount = 0;//calculate the count of mesh that can't be mapped
	int normalEstimation = -1;//Indicates that the nth file is being processed
	bool firstMesh = true;//Used to determine if it is the first successfully obtained mesh
	Eigen::MatrixX3d verticesTarget;
	Eigen::MatrixX3i facesTarget;
	Eigen::VectorXi boundaryTarget;
	Eigen::VectorXi selectConesTarget;
	QProgressBar* bar = ui.convertProgressBar;
	for (const auto& entry : std::filesystem::directory_iterator(depthPath)) {
		if (std::filesystem::is_regular_file(entry) && entry.path().extension() == ".png") {
			normalEstimation++;
			bar->setValue(normalEstimation * 100 / numFiles);
			QApplication::processEvents(); //Update UI response
			if (_stopExecution) {
				QMessageBox::information(nullptr, "Cancel", "User cancel the convertion");
				bar->setValue(0);
				_stopExecution = false;
				return;
			}

			std::string dataName;
			std::vector<std::string> tokens = SplitString(entry.path().filename().string(), ".; _");
			dataName = tokens[1];

			try {
				cv::Mat depthImg = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
				int row = depthImg.rows;
				int col = depthImg.cols;

				std::vector<double> cornerCoordinate = markPoint[normalEstimation];
				int keyPointNum = markPoint[0].size() / 2;

				std::vector<Point_2d> keyPoint;

				bool isDepthMissing = false;
				for (int i = 0; i < keyPointNum; i++) {
					Point_2d nullPoint = { 0,0 };
					keyPoint.emplace_back(nullPoint);
					keyPoint[i].x = std::floor(cornerCoordinate[2 * static_cast<double>(i)]);
					keyPoint[i].y = std::floor(cornerCoordinate[2 * static_cast<double>(i) + 1]);
					if (depthImg.at<uint16_t>(keyPoint[i].y, keyPoint[i].x) == 0) {
						outputInfo->append(("The depth of the key point of file " + entry.path().filename().string() + " is missing.").c_str());
						canNotMapCount++;
						isDepthMissing = true;
						break;
					}
				}
				if (isDepthMissing)
					continue;//Process the next image
				bool isKeyDevia = false;
				for (int i = 0; i < keyPointNum; i++) {
					if (likelihood[normalEstimation][i] < 0.95) {
						outputInfo->append(("The keyPoint of file " + entry.path().filename().string() + " is deviation.").c_str());
						canNotMapCount++;
						isKeyDevia = true;
						break;
					}
				}
				if (isKeyDevia)
					continue;//Process the next image

				// Convert to 3D point
				Point_2d p;
				pcl::PointCloud<pcl::PointXYZ>::Ptr originalVertices(new pcl::PointCloud<pcl::PointXYZ>);
				if (!isConcavePolygon(keyPoint)) {// the polygon enclosed by key points is a convex polygon
					for (int i = 0; i < col; i++) {
						for (int j = 0; j < row; j++) {
							p.x = i;
							p.y = j;
							if (isPointInConvexPolygon(keyPoint, p)) {
								double depth = depthImg.at<uint16_t>(j, i);

								pcl::PointXYZ point;
								point.x = (j - cx) * depth / fx;
								point.y = (i - cy) * depth / fy;
								point.z = depth;
								originalVertices->points.push_back(point);
							}
						}
					}
				}
				else {// the polygon enclosed by key points is a concave polygon
					for (int i = 0; i < col; i++) {
						for (int j = 0; j < row; j++) {
							p.x = i;
							p.y = j;
							if (isPointInConcavePolygon(keyPoint, p)) {
								double depth = depthImg.at<uint16_t>(j, i);

								pcl::PointXYZ point;
								point.x = (j - cx) * depth / fx;
								point.y = (i - cy) * depth / fy;
								point.z = depth;
								originalVertices->points.push_back(point);
							}
						}
					}
				}

				//add the keyPoint
				for (int i = 0; i < keyPointNum; i++) {
					double depth = depthImg.at<uint16_t>(keyPoint[i].y, keyPoint[i].x);

					pcl::PointXYZ point;
					point.x = (keyPoint[i].y - cx) * depth / fx;
					point.y = (keyPoint[i].x - cy) * depth / fy;
					point.z = depth;
					originalVertices->points.push_back(point);
				}
				originalVertices->width = originalVertices->points.size();
				originalVertices->height = 1;

				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutlierRemoved(new pcl::PointCloud<pcl::PointXYZ>);
				originalVertices = TranslateAndNormalizePointCloud(originalVertices);
				pcl::StatisticalOutlierRemoval<pcl::PointXYZ> statisticalOutlierRemoval;
				statisticalOutlierRemoval.setInputCloud(originalVertices);
				statisticalOutlierRemoval.setMeanK(50);
				statisticalOutlierRemoval.setStddevMulThresh(1.0);
				statisticalOutlierRemoval.filter(*cloudOutlierRemoved);

				pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDownsampled(new pcl::PointCloud<pcl::PointXYZ>);
				int downLevel = ui.downLvSpinBox->value();
				if (downLevel != 0) {
					pcl::VoxelGrid<pcl::PointXYZ> downSampled;
					downSampled.setInputCloud(cloudOutlierRemoved);
					double leaf = 0.0001 * pow(2, (downLevel - 1));
					downSampled.setLeafSize(leaf, leaf, leaf);
					downSampled.filter(*cloudDownsampled);
				}

				if (_access(pcPath.c_str(), 0) != 0)
					if (_mkdir(pcPath.c_str()) != 0) {
						QMessageBox::warning(nullptr, "Creaet folder error", ("Can't create folder" + pcPath).c_str());
						return;
					}

				std::string pcFileName = pcPath + "\\pc_" + dataName + ".ply";
				pcl::io::savePLYFile(pcFileName, *cloudDownsampled);

				//Reconstruct mesh
				pcl::PointCloud<pcl::PointNormal>::Ptr cloudWithNormal(new pcl::PointCloud<pcl::PointNormal>);
				pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
				pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>);
				pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

				tree->setInputCloud(cloudDownsampled);
				normalEstimation.setInputCloud(cloudDownsampled);
				normalEstimation.setSearchMethod(tree);
				normalEstimation.setKSearch(20);
				normalEstimation.setViewPoint(0, 0, DBL_MAX);//Set viewpoint at infinity
				normalEstimation.compute(*normal);
				pcl::concatenateFields(*cloudDownsampled, *normal, *cloudWithNormal);

				pcl::Poisson<pcl::PointNormal>  poisson;
				poisson.setInputCloud(cloudWithNormal);
				pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
				poisson.reconstruct(*mesh);

				if (_access(meshPath.c_str(), 0) != 0)
					if (_mkdir(meshPath.c_str()) != 0) {
						QMessageBox::warning(nullptr, "Creaet folder error", ("Can't create folder" + meshPath).c_str());
						return;
					}

				pcl::PointCloud<pcl::PointXYZ>::Ptr verticesOriginal(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::fromPCLPointCloud2(mesh->cloud, *verticesOriginal);
				std::vector<pcl::Vertices> facesOriginal;
				facesOriginal.assign(mesh->polygons.begin(), mesh->polygons.end());
				std::vector<double> verticesX, verticesY, verticesZ;
				for (const pcl::PointXYZ& point : verticesOriginal->points) {
					verticesX.emplace_back(point.x);
					verticesY.emplace_back(point.y);
					verticesZ.emplace_back(point.z);
				}

				//extract keypoints in 3D space
				pcl::PointCloud<pcl::PointXYZ>::Ptr verticesKeyPointI(new pcl::PointCloud<pcl::PointXYZ>);
				std::vector<double> verticesXKeyPoint, verticesYKeyPoint, verticesZKeyPoint;
				for (int i = 0; i < keyPointNum; i++) {
					pcl::PointXYZ point;
					point.x = originalVertices->at(originalVertices->size() - keyPointNum + i).x;
					point.y = originalVertices->at(originalVertices->size() - keyPointNum + i).y;
					point.z = originalVertices->at(originalVertices->size() - keyPointNum + i).z;
					verticesXKeyPoint.emplace_back(point.x);
					verticesYKeyPoint.emplace_back(point.y);
					verticesZKeyPoint.emplace_back(point.z);
					verticesKeyPointI->points.push_back(point);
				}

				Eigen::Map<Eigen::VectorXd> fitVecticesX(verticesXKeyPoint.data(), verticesXKeyPoint.size());
				Eigen::Map<Eigen::VectorXd> fitVecticesY(verticesYKeyPoint.data(), verticesYKeyPoint.size());
				Eigen::Map<Eigen::VectorXd> fitVecticesZ(verticesZKeyPoint.data(), verticesZKeyPoint.size());

				//Fit a plane with key points
				Eigen::MatrixXd fitPlaneCoefficient(keyPointNum, 3);
				fitPlaneCoefficient << fitVecticesX, fitVecticesY, Eigen::VectorXd::Ones(keyPointNum);

				////Using the Least Squares Method to Solve the Coefficients of a Plane Equation
				Eigen::VectorXd fittedPlaneCoefficient = fitPlaneCoefficient.colPivHouseholderQr().solve(fitVecticesZ);
				std::vector<int> preserveVerticesIndex;
				if (!isConcavePolygon(keyPoint)) {// the polygon enclosed by key points is a convex polygon
					double N1 = fittedPlaneCoefficient(0);
					double N2 = fittedPlaneCoefficient(1);
					double N3 = -1.0;
					Eigen::VectorXd A(keyPointNum);
					Eigen::VectorXd B(keyPointNum);
					Eigen::VectorXd C(keyPointNum);
					Eigen::VectorXd depthImg(keyPointNum);

					//Solve the coefficients of planes
					for (int i = 0; i < keyPointNum; i++) {
						Eigen::Vector4d ABCD;
						ABCD = SolveSurface(verticesKeyPointI->at(i), verticesKeyPointI->at((i + 1) % keyPointNum), N1, N2, N3);
						A[i] = ABCD[0];
						B[i] = ABCD[1];
						C[i] = ABCD[2];
						depthImg[i] = ABCD[3];
					}

					std::vector<std::vector<int>> preserveVerticesIndexI;
					preserveVerticesIndexI.resize(keyPointNum);

					// Find the interior point where three planes intersect
					for (int i = 0; i < keyPointNum; i++) {
						//Using key points that are not on the plane as discriminant points
						if (A((i + 1) % keyPointNum) * verticesKeyPointI->at(i).x + B((i + 1) % keyPointNum) * verticesKeyPointI->at(i).y
							+ C((i + 1) % keyPointNum) * verticesKeyPointI->at(i).z + depthImg((i + 1) % keyPointNum) <= 0) {
							for (int j = 0; j < verticesOriginal->size(); j++) {
								if (A((i + 1) % keyPointNum) * verticesOriginal->at(j).x + B((i + 1) % keyPointNum) * verticesOriginal->at(j).y
									+ C((i + 1) % keyPointNum) * verticesOriginal->at(j).z + depthImg((i + 1) % keyPointNum) <= 0)
									preserveVerticesIndexI[i].emplace_back(j);
							}
						}
						else {
							for (int j = 0; j < verticesOriginal->size(); j++) {
								if (A((i + 1) % keyPointNum) * verticesOriginal->at(j).x + B((i + 1) % keyPointNum) * verticesOriginal->at(j).y
									+ C((i + 1) % keyPointNum) * verticesOriginal->at(j).z + depthImg((i + 1) % keyPointNum) >= 0)
									preserveVerticesIndexI[i].emplace_back(j);
							}
						}
						if (i == 0) {
							preserveVerticesIndex = preserveVerticesIndexI[i];
						}
						else {
							std::sort(preserveVerticesIndexI[i].begin(), preserveVerticesIndexI[i].end());
							std::sort(preserveVerticesIndex.begin(), preserveVerticesIndex.end());
							auto it = std::set_intersection(preserveVerticesIndexI[i].begin(), preserveVerticesIndexI[i].end(),
								preserveVerticesIndex.begin(), preserveVerticesIndex.end(), preserveVerticesIndex.begin());
							preserveVerticesIndex.resize(it - preserveVerticesIndex.begin());
						}
					}
				}
				else {// the polygon enclosed by key points is a concave polygon
					// Map the point set to the fitted plane,Ax+By+Cz+depthImg=0 A=fittedPlaneCoefficient(1),
					// B=fittedPlaneCoefficient(2),C=-1,depthImg=fittedPlaneCoefficient(3)
					Eigen::VectorXd verticesXMap, verticesYMap, verticesZMap;
					for (int i = 0; i < verticesX.size(); i++) {
						verticesXMap[i] = (verticesX[i] - fittedPlaneCoefficient[0] * (fittedPlaneCoefficient[0] * verticesX[i] +
							fittedPlaneCoefficient[1] * verticesY[i] - verticesZ[i] + fittedPlaneCoefficient[2]) /
							(pow(fittedPlaneCoefficient[0], 2) + pow(fittedPlaneCoefficient[1], 2) + 1));
						verticesYMap[i] = (verticesY[i] - fittedPlaneCoefficient[1] * (fittedPlaneCoefficient[0] * verticesX[i] +
							fittedPlaneCoefficient[1] * verticesY[i] - verticesZ[i] + fittedPlaneCoefficient[2]) /
							(pow(fittedPlaneCoefficient[0], 2) + pow(fittedPlaneCoefficient[1], 2) + 1));
						verticesZMap[i] = (verticesZ[i] + (fittedPlaneCoefficient[0] * verticesX[i] + fittedPlaneCoefficient[1] *
							verticesY[i] - verticesZ[i] + fittedPlaneCoefficient[2]) / (pow(fittedPlaneCoefficient[0], 2) +
								pow(fittedPlaneCoefficient[1], 2) + 1));
					}
					Eigen::VectorXd verticesXKeyPointMap, verticesYKeyPointMap, verticesZKeyPointMap;
					for (int i = 0; i < verticesXKeyPoint.size(); i++) {
						verticesXKeyPointMap[i] = (verticesXKeyPoint[i] - fittedPlaneCoefficient[0] * (fittedPlaneCoefficient[0] *
							verticesXKeyPoint[i] + fittedPlaneCoefficient[1] * verticesYKeyPoint[i] - verticesZKeyPoint[i] +
							fittedPlaneCoefficient[2]) / (pow(fittedPlaneCoefficient[0], 2) + pow(fittedPlaneCoefficient[1], 2) + 1));
						verticesYKeyPointMap[i] = (verticesYKeyPoint[i] - fittedPlaneCoefficient[1] * (fittedPlaneCoefficient[0] *
							verticesXKeyPoint[i] + fittedPlaneCoefficient[1] * verticesYKeyPoint[i] - verticesZKeyPoint[i] +
							fittedPlaneCoefficient[2]) / (pow(fittedPlaneCoefficient[0], 2) + pow(fittedPlaneCoefficient[1], 2) + 1));
						verticesZKeyPointMap[i] = (verticesZKeyPoint[i] + (fittedPlaneCoefficient[0] * verticesXKeyPoint[i] +
							fittedPlaneCoefficient[1] * verticesYKeyPoint[i] - verticesZKeyPoint[i] + fittedPlaneCoefficient[2]) /
							(pow(fittedPlaneCoefficient[0], 2) + pow(fittedPlaneCoefficient[1], 2) + 1));
					}

					//Rotate point to xOy plane
					Eigen::Vector3d fitPlaneNormal = { fittedPlaneCoefficient[0],fittedPlaneCoefficient[1],-1 };
					Eigen::Vector3d xOy_n = { 0,0,1 };
					double dot = fitPlaneNormal.dot(xOy_n);
					double normFit = fitPlaneNormal.norm();
					double normxOy = xOy_n.norm();
					double theta = std::acos(dot / (normFit * normxOy));
					Eigen::Vector3d rotateAxis = fitPlaneNormal.cross(xOy_n) / (fitPlaneNormal.cross(xOy_n)).norm();
					Eigen::MatrixXd verticesMap;
					verticesMap << verticesXMap, verticesYMap, verticesZMap;
					Eigen::MatrixXd verticesKeyPointMap;
					verticesKeyPointMap << verticesXKeyPointMap, verticesYKeyPointMap, verticesZKeyPointMap;
					Eigen::MatrixXd verticesRotated = RotatePoint(verticesMap, theta, rotateAxis);
					Eigen::MatrixXd verticesKeyPointRotated = RotatePoint(verticesKeyPointMap, theta, rotateAxis);

					//Map the rotate point to 2D space
					std::vector<Point_2d> verticesXYKeyPointRotated;
					for (int i = 0; i < keyPointNum; i++) {
						Point_2d p;
						p.x = verticesKeyPointRotated(i, 0);
						p.y = verticesKeyPointRotated(i, 1);
						verticesXYKeyPointRotated.emplace_back(p);
					}
					for (int i = 0; i < verticesX.size(); i++) {
						Point_2d p;
						p.x = verticesRotated(i, 0);
						p.y = verticesRotated(i, 1);
						if (isPointInConcavePolygon(verticesXYKeyPointRotated, p))
							preserveVerticesIndex.emplace_back(i);
					}
				}
				RemoveVerticesAndFacesByValue(mesh, preserveVerticesIndex, facesOriginal);

				// Mapping to obtain meshes with the same topology
				if (firstMesh) {
					verticesTarget.resize(mesh->cloud.width, 3);
					facesTarget.resize(mesh->polygons.size(), 3);

					pcl::PointCloud<pcl::PointXYZ>::Ptr pcTarget(new pcl::PointCloud<pcl::PointXYZ>);
					pcl::fromPCLPointCloud2(mesh->cloud, *pcTarget);
					for (int i = 0; i < pcTarget->width; i++) {
						verticesTarget(i, 0) = pcTarget->points[i].x;
						verticesTarget(i, 1) = pcTarget->points[i].y;
						verticesTarget(i, 2) = pcTarget->points[i].z;
					}
					for (int i = 0; i < mesh->polygons.size(); i++) {
						pcl::Vertices::Ptr tempVertices(new pcl::Vertices);
						*tempVertices = mesh->polygons[i];
						facesTarget(i, 0) = (*tempVertices).vertices[0];
						facesTarget(i, 1) = (*tempVertices).vertices[1];
						facesTarget(i, 2) = (*tempVertices).vertices[2];
					}

					Surface_mesh cgalMesh;
					cgalMesh.clear();
					for (const auto& point : pcTarget->points)
						cgalMesh.add_vertex(Point_3(point.x, point.y, point.z));
					for (const auto& polygon : mesh->polygons)
						cgalMesh.add_face(Vertex_index(polygon.vertices[0]), Vertex_index(polygon.vertices[1]),
							Vertex_index(polygon.vertices[2]));
					Surface_mesh::Halfedge_range halfEdges = cgalMesh.halfedges();
					//get the unorder free boundary
					for (auto it = halfEdges.begin(); it != halfEdges.end(); it++) {
						if (cgalMesh.is_border(*it)) {
							boundaryTarget.conservativeResize(boundaryTarget.size() + 1);
							Vertex_index boundaryIndex = cgalMesh.source(*it);
							boundaryTarget[boundaryTarget.size() - 1] = boundaryIndex;
						}
					}

					std::unordered_map<Vertex_index, Vertex_index> nextBoundaryVertex;
					for (auto it = halfEdges.begin(); it != halfEdges.end(); it++)
						if (cgalMesh.is_border(*it)) {
							Vertex_index boundaryIndex = cgalMesh.source(*it);
							Vertex_index nextBoundaryIndex = cgalMesh.target(*it);
							nextBoundaryVertex[boundaryIndex] = nextBoundaryIndex;
						}

					Vertex_index boundaryTargetStart = Vertex_index(boundaryTarget[0]);
					size_t boundaryTargetIndex = 0;
					bool firstError = false;
					while (true) {
						boundaryTarget[boundaryTargetIndex] = boundaryTargetStart;
						boundaryTargetIndex++;
						auto it = nextBoundaryVertex.find(boundaryTargetStart);
						if (boundaryTargetIndex == boundaryTarget.size())
							break;

						if (it == nextBoundaryVertex.end()) {
							firstError = true;
							break;
						}
						boundaryTargetStart = it->second;
						nextBoundaryVertex.erase(it);
					}
					if (firstError)
						continue;

					//Using the position of the first keyPoint as the starting index
					int nearestKeyPoint1Index = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(0));
					Eigen::VectorXi nearestKeyPoint1Head = boundaryTarget.head(nearestKeyPoint1Index);
					boundaryTarget.segment(0, boundaryTarget.size() - nearestKeyPoint1Index) =
						boundaryTarget.segment(nearestKeyPoint1Index, boundaryTarget.size() - nearestKeyPoint1Index);
					boundaryTarget.segment(boundaryTarget.size() - nearestKeyPoint1Index, nearestKeyPoint1Index) =
						nearestKeyPoint1Head;

					Eigen::VectorXi modelKeyPointIndex(keyPointNum);
					for (int i = 0; i < keyPointNum; i++)
						modelKeyPointIndex[i] = boundaryTarget[Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(i))];
					std::ofstream keyPointIdxFile(std::string(mainPath) + "/" + _projectName.toStdString() + "/modelKeyPointIndex.txt");
					if (keyPointIdxFile.is_open()) {
						for (int i = 0; i < modelKeyPointIndex.size(); i++)
							keyPointIdxFile << modelKeyPointIndex[i] << std::endl;
					}
					keyPointIdxFile.close();

					//search the corners around the egde
					if (keyPointNum == 3) {
						int nearestVertices1IdxTarget = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(0));
						int nearestVertices2IdxTarget = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(1));
						int nearestVertices3IdxTarget = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(2));
						selectConesTarget.resize(3);
						selectConesTarget[0] = nearestVertices1IdxTarget;
						selectConesTarget[1] = nearestVertices2IdxTarget;
						selectConesTarget[2] = nearestVertices3IdxTarget;
					}
					else {
						double interval = keyPointNum / 4.0;
						int nearestVertices1IdxTarget = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(round(interval * 1 - 1)));
						int nearestVertices2IdxTarget = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(round(interval * 2 - 1)));
						int nearestVertices3IdxTarget = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(round(interval * 3 - 1)));
						int nearestVertices4IdxTarget = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(round(interval * 4 - 1)));
						selectConesTarget.resize(4);
						selectConesTarget[0] = nearestVertices1IdxTarget;
						selectConesTarget[1] = nearestVertices2IdxTarget;
						selectConesTarget[2] = nearestVertices3IdxTarget;
						selectConesTarget[3] = nearestVertices4IdxTarget;
					}
					std::sort(selectConesTarget.data(), selectConesTarget.data() + selectConesTarget.size());

					firstMesh = false;

					//save mesh
					std::string meshName = meshPath + "\\mesh_" + dataName + ".ply";
					pcl::io::savePLYFile(meshName, *mesh);
				}
				else { // not first mesh
					Eigen::MatrixX3d verticesSource(mesh->cloud.width, 3);
					Eigen::MatrixX3i facesSource(mesh->polygons.size(), 3);
					Eigen::VectorXi boundarySource;
					Eigen::VectorXi selectConesSource;

					pcl::PointCloud<pcl::PointXYZ>::Ptr pcSource(new pcl::PointCloud<pcl::PointXYZ>);
					pcl::fromPCLPointCloud2(mesh->cloud, *pcSource);

					for (int i = 0; i < pcSource->width; i++) {
						verticesSource(i, 0) = pcSource->points[i].x;
						verticesSource(i, 1) = pcSource->points[i].y;
						verticesSource(i, 2) = pcSource->points[i].z;
					}

					for (int i = 0; i < mesh->polygons.size(); i++) {
						pcl::Vertices::Ptr tempVertices(new pcl::Vertices);
						*tempVertices = mesh->polygons[i];
						facesSource(i, 0) = (*tempVertices).vertices[0];
						facesSource(i, 1) = (*tempVertices).vertices[1];
						facesSource(i, 2) = (*tempVertices).vertices[2];
					}

					Surface_mesh cgalMesh;
					cgalMesh.clear();
					for (const auto& point : pcSource->points)
						cgalMesh.add_vertex(Point_3(point.x, point.y, point.z));
					for (const auto& polygon : mesh->polygons)
						cgalMesh.add_face(Vertex_index(polygon.vertices[0]), Vertex_index(polygon.vertices[1]),
							Vertex_index(polygon.vertices[2]));
					Surface_mesh::Halfedge_range halfEdges = cgalMesh.halfedges();

					// get the free boundary
					for (auto it = halfEdges.begin(); it != halfEdges.end(); it++)
						if (cgalMesh.is_border(*it)) {
							boundarySource.conservativeResize(boundarySource.size() + 1);
							Vertex_index boundaryIndex = cgalMesh.source(*it);
							boundarySource[boundarySource.size() - 1] = boundaryIndex;
						}

					std::unordered_map<Vertex_index, Vertex_index> nextBoundaryVertex;
					for (auto it = halfEdges.begin(); it != halfEdges.end(); it++)
						if (cgalMesh.is_border(*it)) {
							Vertex_index boundaryIndex = cgalMesh.source(*it);
							Vertex_index nextBoundaryIndex = cgalMesh.target(*it);
							nextBoundaryVertex[boundaryIndex] = nextBoundaryIndex;
						}

					Vertex_index boundarySourceStart = Vertex_index(boundarySource[0]);
					size_t index = 0;
					while (true) {
						boundarySource[index] = boundarySourceStart;
						index++;
						auto it = nextBoundaryVertex.find(boundarySourceStart);
						if (index == boundarySource.size() || it == nextBoundaryVertex.end()) {
							break;
						}
						boundarySourceStart = it->second;
						nextBoundaryVertex.erase(it);
					}

					//Using the position of the first keyPoint as the starting index
					int nearestKeyPoint1Index = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(0));
					Eigen::VectorXi nearestKeyPoint1Head = boundarySource.head(nearestKeyPoint1Index);
					boundarySource.segment(0, boundarySource.size() - nearestKeyPoint1Index) =
						boundarySource.segment(nearestKeyPoint1Index, boundarySource.size() - nearestKeyPoint1Index);
					boundarySource.segment(boundarySource.size() - nearestKeyPoint1Index, nearestKeyPoint1Index) =
						nearestKeyPoint1Head;

					Eigen::VectorXi modelKeyPointIndex(keyPointNum);
					for (int i = 0; i < keyPointNum; i++)
						modelKeyPointIndex[i] = boundarySource[Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(i))];

					//search the corners around the egde
					if (keyPointNum == 3) {
						int nearestVertices1IdxSource = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(0));
						int nearestVertices2IdxSource = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(1));
						int nearestVertices3IdxSource = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(2));
						selectConesSource.resize(3);
						selectConesSource[0] = nearestVertices1IdxSource;
						selectConesSource[1] = nearestVertices2IdxSource;
						selectConesSource[2] = nearestVertices3IdxSource;
					}
					else {
						double interval = keyPointNum / 4.0;
						int nearestVertices1IdxSource = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(round(interval * 1 - 1)));
						int nearestVertices2IdxSource = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(round(interval * 2 - 1)));
						int nearestVertices3IdxSource = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(round(interval * 3 - 1)));
						int nearestVertices4IdxSource = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(round(interval * 4 - 1)));
						selectConesSource.resize(4);
						selectConesSource[0] = nearestVertices1IdxSource;
						selectConesSource[1] = nearestVertices2IdxSource;
						selectConesSource[2] = nearestVertices3IdxSource;
						selectConesSource[3] = nearestVertices4IdxSource;
					}
					std::sort(selectConesSource.data(), selectConesSource.data() + selectConesSource.size());

					Eigen::MatrixX3d verticesMapped(verticesTarget.rows(), verticesTarget.cols());
					OrbifoldTutteEmbedding orbifold;
					if (!orbifold.disk_map(verticesTarget, facesTarget, boundaryTarget, verticesSource, facesSource, boundarySource,
						selectConesTarget, selectConesSource, verticesMapped)) {
						outputInfo->append((dataName + " can not be map.").c_str());
						canNotMapCount++;
						continue;
					}

					pcl::PointCloud<pcl::PointXYZ>::Ptr pcMapped(new pcl::PointCloud<pcl::PointXYZ>);
					pcMapped->resize(verticesMapped.rows());
					for (int i = 0; i < verticesMapped.rows(); i++) {
						pcl::PointXYZ mapPoint;
						mapPoint.x = verticesMapped(i, 0);
						mapPoint.y = verticesMapped(i, 1);
						mapPoint.z = verticesMapped(i, 2);
						pcMapped->points[i] = mapPoint;
					}

					pcl::PolygonMesh::Ptr meshMapped(new pcl::PolygonMesh);
					meshMapped->polygons.resize(facesTarget.rows());
					for (int i = 0; i < facesTarget.rows(); i++) {
						pcl::Vertices mapVertices;
						mapVertices.vertices.push_back(facesTarget(i, 0));
						mapVertices.vertices.push_back(facesTarget(i, 1));
						mapVertices.vertices.push_back(facesTarget(i, 2));
						meshMapped->polygons[i] = mapVertices;
					}
					pcl::toPCLPointCloud2(*pcMapped, meshMapped->cloud);

					std::string meshName = meshPath + "\\mesh_" + dataName + ".ply";
					pcl::io::savePLYFile(meshName, *meshMapped);
				}
			}
			catch (const std::exception& e) {
				canNotMapCount++;
				outputInfo->append((dataName + " can not be map.").c_str());
				continue;
			}
		}
	}
	bar->setValue(100);
	QMessageBox::information(nullptr, "Complete", "Convert complete.");

	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	outputInfo->append(("Cost total " + std::to_string(duration.count() * 0.001) + " seconds").c_str());
	outputInfo->append(("Total " + std::to_string(canNotMapCount) + " mesh reconstruction fail").c_str());
}


void LShapeAnalyser_single::loadTruth() {
	char mainPath[_MAX_PATH];
	if (!_getcwd(mainPath, _MAX_PATH)) {
		QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
		return;
	}
	std::string truthFilePath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\Truth.csv";
	if (_access(truthFilePath.c_str(), 0) != 0) {
		QMessageBox::warning(nullptr, "Not file exist", "Not file exist");
		return;
	}
	else {
		QTableWidget* truthTable = ui.truthTable;
		// clear the old
		while (truthTable->rowCount() > 0)
			truthTable->removeRow(0);

		QString truthFileName = truthFilePath.c_str();
		QFile file(truthFileName);
		if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
			QMessageBox::warning(nullptr, "Can't open file", "Can't open file");
			return;
		}
		QTextStream in(&file);
		QString headerLine = in.readLine();
		QStringList headerFields = headerLine.split(",");
		int columnCount = headerFields.size();
		truthTable->setColumnCount(columnCount);
		truthTable->setHorizontalHeaderLabels(headerFields);

		int row = 0;
		while (!in.atEnd()) {
			QString line = in.readLine();
			QStringList fields = line.split(",");

			truthTable->insertRow(row);

			for (int column = 0; column < columnCount; ++column) {
				QTableWidgetItem* item = new QTableWidgetItem(fields[column]);
				if (item)
					truthTable->setItem(row, column, item);
			}
			row++;
		}
		file.close();
	}
}


void LShapeAnalyser_single::resetTruth() {
	QMessageBox::StandardButton reply;
	reply = QMessageBox::question(nullptr, "Confirm", "Reset file name and truth value?", QMessageBox::Yes | QMessageBox::No);
	if (reply == QMessageBox::No)
		return;
	char mainPath[_MAX_PATH];
	if (!_getcwd(mainPath, _MAX_PATH)) {
		QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
		return;
	}
	std::string truthFilePath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\Truth.csv";
	if (_access(truthFilePath.c_str(), 0) == 0)//remove old
		remove(truthFilePath.c_str());
	if (_access(truthFilePath.c_str(), 0) != 0) {
		std::string depthPath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\depth";
		std::vector<std::string> truthFileName;
		std::vector<double> truthValue;
		_finddatai64_t fileInfo;
		intptr_t fileHandel = 0;
		if ((fileHandel = _findfirst64((depthPath + "\\*.png").c_str(), &fileInfo)) != -1) {
			do {
				truthFileName.emplace_back(fileInfo.name);
				truthValue.emplace_back(0.0f);
			} while (_findnext64(fileHandel, &fileInfo) == 0);
		}
		std::ofstream truthValueFile(truthFilePath);
		if (!truthValueFile.is_open()) {
			QMessageBox::warning(nullptr, "File open error", "Can't open truthValueFile");
			return;
		}
		truthValueFile << "truthFileName,truth" << std::endl;
		for (int i = 0; i < truthFileName.size(); i++) {
			truthValueFile << truthFileName[i] << "," << truthValue[i] << std::endl;
		}
		truthValueFile.close();
	}

	QTableWidget* truthTable = ui.truthTable;
	// clear the old
	while (truthTable->rowCount() > 0)
		truthTable->removeRow(0);

	QString truthFileName = truthFilePath.c_str();
	QFile QTruthValueFile(truthFileName);
	if (!QTruthValueFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
		QMessageBox::warning(nullptr, "Can't open file", "Can't open file");
		return;
	}
	QTextStream in(&QTruthValueFile);
	QString headerLine = in.readLine();
	QStringList headerFields = headerLine.split(",");
	int columnCount = headerFields.size();
	truthTable->setColumnCount(columnCount);
	truthTable->setHorizontalHeaderLabels(headerFields);

	int row = 0;
	while (!in.atEnd()) {
		QString line = in.readLine();
		QStringList fields = line.split(",");

		truthTable->insertRow(row);

		for (int column = 0; column < columnCount; ++column) {
			QTableWidgetItem* item = new QTableWidgetItem(fields[column]);
			truthTable->setItem(row, column, item);
			column++;
		}
		row++;
	}
	QTruthValueFile.close();
}


void LShapeAnalyser_single::importTruth() {
	QString filePath = QFileDialog::getOpenFileName(nullptr, "Select truth value file", "", "ALL Files (*.*)");
	if (!filePath.isEmpty()) {
		char mainPath[_MAX_PATH];
		if (!_getcwd(mainPath, _MAX_PATH)) {//Get the main path
			QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
			return;
		}

		// calculate the file numbers
		QTableWidget* truthTable = ui.truthTable;
		int fileNum = truthTable->rowCount();
		if (fileNum == 0) {
			QMessageBox::warning(nullptr, "Not data load", "Data has not been loaded yet");
			return;
		}

		std::ifstream file(filePath.toStdString());
		if (!file.is_open()) {
			QMessageBox::warning(nullptr, "File open error", "Can't open file");
			return;
		}
		std::vector<std::vector<std::string>> tableData;
		std::string line;
		while (std::getline(file, line)) {
			std::vector<std::string> row;
			size_t pos = 0;
			std::string cell;

			while ((pos = line.find('\t')) != std::string::npos || (pos = line.find(',')) != std::string::npos) {
				cell = line.substr(0, pos);
				line.erase(0, pos + 1);//Remove cells that have already been read
				row.push_back(cell);
			}

			row.push_back(line);//Add last cell
			tableData.push_back(row);
		}

		file.close();

		size_t rowCount = tableData.size();
		size_t columnSize = tableData[0].size();
		if (columnSize != 1) {
			QMessageBox::warning(nullptr, "Column inconsistent", "There can only be one column of file data");
			return;
		}
		if (rowCount != fileNum) {
			QMessageBox::warning(nullptr, "Inconsistent number of files", "Inconsistent number of files and import value");
			return;
		}
		for (int i = 0; i < fileNum; i++) {
			QTableWidgetItem* item = new QTableWidgetItem(tableData[i].at(0).c_str());
			truthTable->setItem(i, truthTable->columnCount() - 1, item);
		}
		QMessageBox::information(nullptr, "Import successfully", "Import successfully");
	}
	else {
		return;
	}
}


void LShapeAnalyser_single::saveTruth() {
	char mainPath[_MAX_PATH];
	if (!_getcwd(mainPath, _MAX_PATH)) {
		QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
		return;
	}
	std::string truthFilePath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\Truth.csv";
	if (_access(truthFilePath.c_str(), 0) == 0)//remove old
		remove(truthFilePath.c_str());
	std::ofstream file(truthFilePath);
	if (!file.is_open()) {
		QMessageBox::warning(nullptr, "File open error", "Can't open file");
		return;
	}
	QTableWidget* truthTable = ui.truthTable;
	for (int i = 0; i < truthTable->columnCount(); i++) {
		QTableWidgetItem* headerItem = truthTable->horizontalHeaderItem(i);
		if (headerItem)
			file << headerItem->text().toStdString().c_str();
		if (i < truthTable->columnCount() - 1)
			file << ",";
	}
	file << std::endl;

	for (int i = 0; i < truthTable->rowCount(); i++) {
		for (int j = 0; j < truthTable->columnCount(); j++) {
			if (truthTable->item(i, j)) {
				file << truthTable->item(i, j)->text().toStdString().c_str();
				if (j < truthTable->columnCount() - 1)
					file << ",";
			}
		}
		file << std::endl;
	}
	file.close();
	QMessageBox::information(nullptr, "Save successfully", "Save successfully");
}


void LShapeAnalyser_single::startEvaluate() {
	// Pre-processing data(Train)
	char mainPath[_MAX_PATH];
	if (!_getcwd(mainPath, _MAX_PATH)) {
		QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
		return;
	}
	std::string meshPath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\mesh";
	pcl::PolygonMesh::Ptr trainMesh(new pcl::PolygonMesh);//average shape variable

	_finddatai64_t fileInfo;
	intptr_t fileHandel = 0;
	if (_access(meshPath.c_str(), 0) != 0 || (fileHandel = _findfirst64((meshPath + "\\*.ply").c_str(), &fileInfo)) == -1) {
		QMessageBox::warning(nullptr, "Mesh data not exist", "Mesh data does not exist, please reconstruct the mesh firstly");
		return;
	}
	else {
		pcl::io::loadPLYFile((meshPath + "\\" + fileInfo.name).c_str(), *trainMesh);
	}

	// Calculate the number of files
	int fileNumber = 0;
	for (const auto& entry : std::filesystem::directory_iterator(meshPath))
		if (std::filesystem::is_regular_file(entry) && entry.path().extension() == ".ply")
			fileNumber++;

	std::vector<int> fileIdx;
	for (int i = 0; i < fileNumber; i++)
		fileIdx.emplace_back(i);

	int trainNum = fileNumber * 0.8;//The size of  data set used to train
	int testNum = fileNumber - trainNum; //The size of data set used to test
	QTextEdit* setsText = ui.setsDescribeText;
	setsText->clear();
	setsText->append(("Train set size: " + std::to_string(trainNum) + "\nTest set size: " + std::to_string(testNum)).c_str());

	// Read truth value
	std::string truthFilePath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\Truth.csv";
	if (_access(truthFilePath.c_str(), 0) != 0) {
		QMessageBox::warning(nullptr, "Not file exist", "Truth value file is not exist");
		return;
	}

	std::vector<std::string> remainFile;
	do {
		std::string number = fileInfo.name;
		remainFile.emplace_back(number);
	} while (_findnext64(fileHandel, &fileInfo) == 0);

	std::vector<std::string> truthFileName;
	std::vector<double> truthValue;
	std::ifstream file(truthFilePath);
	if (!file.is_open()) {
		QMessageBox::warning(nullptr, "File open error", "Can't open file Truth.csv");
		return;
	}
	std::string line;
	std::getline(file, line);//get the title row
	while (std::getline(file, line)) {
		std::string cell;
		size_t pos = 0;
		if ((pos = line.find(',')) != std::string::npos) {
			cell = line.substr(0, pos);
			line.erase(0, pos + 1);

			std::string oldSubstring = "depth";
			std::string newSubstring = "mesh";

			size_t pos2 = 0;
			while ((pos2 = cell.find(oldSubstring, pos2)) != std::string::npos) {
				cell.replace(pos2, oldSubstring.length(), newSubstring);
				pos2 += newSubstring.length();
			}
			truthFileName.emplace_back(cell);
		}
		truthValue.emplace_back(std::stod(line));
	}
	file.close();

	for (int i = 0; i < truthFileName.size(); i++) {
		std::string oldStr = "png";
		std::string newStr = "ply";
		size_t pos = 0;
		if ((pos = truthFileName[i].find(oldStr, pos)) != std::string::npos)
			truthFileName[i].replace(pos, oldStr.length(), newStr);
	}

	//Remove parameters corresponding to files that failed reconstruction
	for (int i = truthFileName.size() - 1; i >= 0; i--) {
		auto it = std::find(remainFile.begin(), remainFile.end(), truthFileName[i]);
		if (it == remainFile.end()) {
			truthFileName.erase(truthFileName.begin() + i);
			truthValue.erase(truthValue.begin() + i);
		}
	}

	int timesFold = ui.testTimesSpinBox->value();
	Eigen::VectorXd accuracy(timesFold);
	Eigen::MatrixX4d approximateAccuracy(timesFold, 4);
	QProgressBar* bar = ui.evaluateProgressBar;
	for (int i = 0; i < timesFold; i++) {
		//Determine if the user has interrupted the program
		QApplication::processEvents(); //Update UI response
		if (_stopExecution) {
			QMessageBox::information(nullptr, "Cancel", "User cancel the evaluation");
			bar->setValue(0);
			_stopExecution = false;
			return;
		}
		bar->setValue(i * 100 / timesFold);

		//Obtain the training set randomly from whole set
		std::vector<int> trainIdx = fileIdx;
		std::vector<int> testIdx;
		// Get a random seed
		std::random_device rd;
		std::mt19937 rng(rd());
		// Shuffle the randomVector
		std::shuffle(trainIdx.begin(), trainIdx.end(), rng);
		// Resize the randomVector to contain only normalEstimation elements
		auto startIterator = trainIdx.end() - testNum;
		testIdx.insert(testIdx.end(), startIterator, trainIdx.end());
		trainIdx.resize(trainNum);

		//Obtain the truth value corresponding to the training set
		std::vector<double> truthValueTrain;
		for (int j = 0; j < trainNum; j++)
			truthValueTrain.emplace_back(truthValue[trainIdx[j]]);

		//Set the remaining dataset as the test set
		std::vector<double> truthValueTest;
		for (int j = 0; j < testNum; j++)
			truthValueTest.emplace_back(truthValue[testIdx[j]]);

		//Read the file name of train set and test set, respectively
		std::vector<std::string> trainFileName;
		std::vector<std::string> testFileName;
		for (int j = 0; j < trainNum; j++)
			trainFileName.emplace_back(truthFileName[trainIdx[j]]);
		for (int j = 0; j < testNum; j++)
			testFileName.emplace_back(truthFileName[testIdx[j]]);

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::fromPCLPointCloud2(trainMesh->cloud, *cloud);
		std::vector<pcl::Vertices> polygon = trainMesh->polygons;
		Eigen::MatrixX3d V_0(cloud->size(), 3);
		Eigen::MatrixX3i T_0(polygon.size(), 3);
		for (int j = 0; j < cloud->size(); j++) {
			V_0(j, 0) = cloud->points[j].x;
			V_0(j, 1) = cloud->points[j].y;
			V_0(j, 2) = cloud->points[j].z;
		}
		for (int j = 0; j < polygon.size(); j++) {
			T_0(j, 0) = polygon[j].vertices.at(0);
			T_0(j, 1) = polygon[j].vertices.at(1);
			T_0(j, 2) = polygon[j].vertices.at(2);
		}

		//Translate the target point's center of gravity to the origin
		Eigen::VectorXd centroid = V_0.colwise().mean();
		V_0.rowwise() -= centroid.transpose();
		//normalization
		V_0 /= V_0.norm();

		int verticesNum = V_0.rows();
		std::vector<Eigen::MatrixX3d> V;

		//Translate the source points' center of gravity to the origin and normalization it
		for (int j = 0; j < trainNum; j++) {
			pcl::PolygonMesh::Ptr trainMesh(new pcl::PolygonMesh);
			pcl::io::loadPLYFile(meshPath + "\\" + trainFileName[j], *trainMesh);
			pcl::PointCloud<pcl::PointXYZ>::Ptr trainPc(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::fromPCLPointCloud2(trainMesh->cloud, *trainPc);
			Eigen::MatrixX3d trainVertices(trainPc->size(), 3);
			for (int k = 0; k < trainPc->size(); k++) {
				trainVertices(k, 0) = trainPc->points[k].x;
				trainVertices(k, 1) = trainPc->points[k].y;
				trainVertices(k, 2) = trainPc->points[k].z;
			}
			Eigen::VectorXd centroidVertices = trainVertices.colwise().mean();
			trainVertices.rowwise() -= centroidVertices.transpose();
			trainVertices /= trainVertices.norm();
			V.emplace_back(trainVertices);
		}

		//Alignment
		int iter = 0;
		while (iter < 10) {
			iter++;

			//Align all shapes to the average shape
			for (int j = 0; j < trainNum; j++)
				Procrustes(V_0, V[j]);

			Eigen::MatrixX3d V_0_temp = GenerateAvg(V, trainNum, verticesNum);
			Procrustes(V_0, V_0_temp);
			V_0 = V_0_temp;
		}

		Eigen::VectorXd V_0Centroid = V_0.colwise().mean();
		Eigen::MatrixXd centered = V_0.rowwise() - V_0Centroid.transpose();
		// Calculate Covariance matrix
		Eigen::MatrixXd covarianceMatrix = (centered.adjoint() * centered) / (centered.rows() - 1);

		// Calculate feature vectors
		Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver(covarianceMatrix);
		Eigen::MatrixXcd eigenVectorsComplex = eigenSolver.eigenvectors(); // Complex eigenvector
		Eigen::VectorXd eigenValues = eigenSolver.eigenvalues().real();

		std::vector<int> sortedIndices(eigenValues.size());
		for (int i = 0; i < eigenValues.size(); ++i)
			sortedIndices[i] = i;

		std::sort(sortedIndices.begin(), sortedIndices.end(), [&](int a, int b) {
			return eigenValues(a) > eigenValues(b);
			});

		Eigen::MatrixXd eigenVectors(3, 3);
		for (int i = 0; i < 3; i++)
			eigenVectors.col(i) = eigenVectorsComplex.real().col(sortedIndices[i]);

		Eigen::Matrix3d rotateX;
		rotateX << 1, 0, 0,
			0, -1, 0,
			0, 0, -1;
		Eigen::VectorXd centroidNorm = V_0.colwise().mean();
		V_0 = (V_0.rowwise() - centroidNorm.transpose()) * eigenVectors * rotateX;
		for (int j = 0; j < trainNum; j++)
			V[j] = (V[j].rowwise() - centroidNorm.transpose()) * eigenVectors * rotateX;

		auto minValue = std::min_element(truthValueTrain.begin(), truthValueTrain.end());
		auto maxValue = std::max_element(truthValueTrain.begin(), truthValueTrain.end());
		double minTruth = *minValue;
		double maxTruth = *maxValue;

		//Calculate scores in training sets
		Eigen::VectorXd trainScore(trainNum);
		for (int j = 0; j < trainNum; j++)
			trainScore[j] = EvaluateScore(V_0, V[j], T_0);

		double minScore = trainScore.minCoeff();
		double maxScore = trainScore.maxCoeff();

		//Evaluate scores of all data(Test)
		Eigen::VectorXd testScore(testNum);
		//Calculate scores of all test data
		for (int j = 0; j < testNum; j++) {
			pcl::PolygonMesh::Ptr testMesh(new pcl::PolygonMesh);
			pcl::io::loadPLYFile(meshPath + "\\" + testFileName[j], *testMesh);
			pcl::PointCloud<pcl::PointXYZ>::Ptr testPc(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::fromPCLPointCloud2(testMesh->cloud, *testPc);
			Eigen::MatrixX3d V_S(testPc->size(), 3);
			for (int k = 0; k < testPc->size(); k++) {
				V_S(k, 0) = testPc->points[k].x;
				V_S(k, 1) = testPc->points[k].y;
				V_S(k, 2) = testPc->points[k].z;
			}
			//Translate the target point's center of gravity to the origin
			Eigen::VectorXd V_0Centroid = V_S.colwise().mean();
			V_S.rowwise() -= V_0Centroid.transpose();
			V_S /= V_S.norm();
			Procrustes(V_0, V_S);
			testScore[j] = EvaluateScore(V_0, V_S, T_0);
		}

		//Standardized truth
		double up = ui.upperLimitLineEdit->text().toDouble();
		double low = ui.lowerLimitLineEdit->text().toDouble();
		if (up < low) {
			QMessageBox::warning(nullptr, "limit error", "The lower limit must be less than the upper");
			return;
		}
		double current = low;
		double step = ui.discreIncreLineEdit->text().toDouble();
		Eigen::VectorXd enumValue(int((up - low) / step + 1));
		int idx = 0;
		while (current <= up) {
			enumValue[idx] = current;
			current += step;
			idx++;
		}
		Eigen::VectorXd normScore = (testScore - Eigen::VectorXd::Constant(testScore.size(), minScore)) /
			(maxScore - minScore) * (maxTruth - minTruth) + Eigen::VectorXd::Constant(testScore.size(), minTruth);

		// Calculate accuracy
		int accuracyNum = 0;
		Eigen::Vector4d approximateNum = Eigen::Vector4d::Zero();

		for (int j = 0; j < testNum; j++) {
			// Discretizing Truth
			if (ui.discreIncreLineEdit->text().toDouble() != 0) {
				Eigen::VectorXd neigh = (enumValue - Eigen::VectorXd::Constant(enumValue.size(), normScore[j])).cwiseAbs();
				auto it = std::find(neigh.begin(), neigh.end(), neigh.minCoeff());
				int enumIdx = it - neigh.begin();
				normScore[j] = enumValue[enumIdx];
			}
			if (std::abs(normScore[j] - truthValueTest[j]) <= 0)
				accuracyNum++;

			double disIncre = ui.displayIncreLineEdit->text().toDouble();
			for (size_t k = 0; k < approximateNum.size(); k++)
				if (std::abs(normScore[j] - truthValueTest[j]) <= disIncre * (k + 1))
					approximateNum[k] += 1;
		}
		accuracy[i] = double(accuracyNum) / double(testNum);
		approximateAccuracy.row(i) = approximateNum / double(testNum);
	}
	bar->setValue(100);
	Eigen::VectorXd Avg(5);
	Eigen::VectorXd Min(5);
	Eigen::VectorXd Max(5);
	for (int i = 0; i < 5; i++) {
		if (i == 0) {
			Avg[i] = accuracy.mean() * 100;
			Min[i] = accuracy.minCoeff() * 100;
			Max[i] = accuracy.maxCoeff() * 100;
		}
		else {
			Avg[i] = approximateAccuracy.col(static_cast<int64>(i) - 1).mean() * 100;
			Min[i] = approximateAccuracy.col(static_cast<int64>(i) - 1).minCoeff() * 100;
			Max[i] = approximateAccuracy.col(static_cast<int64>(i) - 1).maxCoeff() * 100;
		}
	}
	QTableWidget* evaluateTable = ui.evaluateAccTable;

	// clear the old itme
	while (evaluateTable->rowCount() > 0)
		evaluateTable->removeRow(0);
	evaluateTable->setColumnCount(4);

	//add the header
	QTableWidgetItem* item1 = new QTableWidgetItem("Error");
	QTableWidgetItem* item2 = new QTableWidgetItem("Avg");
	QTableWidgetItem* item3 = new QTableWidgetItem("Min");
	QTableWidgetItem* item4 = new QTableWidgetItem("Max");
	evaluateTable->setHorizontalHeaderItem(0, item1);
	evaluateTable->setHorizontalHeaderItem(1, item2);
	evaluateTable->setHorizontalHeaderItem(2, item3);
	evaluateTable->setHorizontalHeaderItem(3, item4);

	double disIncre = ui.discreIncreLineEdit->text().toDouble();
	for (int i = 0; i < Avg.size(); i++) {
		evaluateTable->insertRow(i);
		for (int j = 0; j < 4; j++) {
			if (j == 0) {
				QTableWidgetItem* item = new QTableWidgetItem(("Error<=" + std::to_string(disIncre * i)).c_str());
				if (item)
					evaluateTable->setItem(i, j, item);
			}
			else if (j == 1) {
				QTableWidgetItem* item = new QTableWidgetItem(std::to_string(Avg[i]).c_str());
				if (item)
					evaluateTable->setItem(i, j, item);
			}
			else if (j == 2) {
				QTableWidgetItem* item = new QTableWidgetItem(std::to_string(Min[i]).c_str());
				if (item)
					evaluateTable->setItem(i, j, item);
			}
			else {
				QTableWidgetItem* item = new QTableWidgetItem(std::to_string(Max[i]).c_str());
				if (item)
					evaluateTable->setItem(i, j, item);
			}
		}
	}
	QMessageBox::information(nullptr, "Finish", "Evaluation finish.");
}


void LShapeAnalyser_single::train() {
	// Pre-processing data
	char mainPath[_MAX_PATH];
	if (!_getcwd(mainPath, _MAX_PATH)) {
		QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
		return;
	}
	std::string meshPath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\mesh";
	pcl::PolygonMesh::Ptr avgMesh(new pcl::PolygonMesh);//average shape variable

	_finddatai64_t fileInfo;
	intptr_t fileHandel = 0;
	if (_access(meshPath.c_str(), 0) != 0 || (fileHandel = _findfirst64((meshPath + "\\*.ply").c_str(), &fileInfo)) == -1) {
		QMessageBox::warning(nullptr, "Mesh data not exist", "Mesh data does not exist, please reconstruct the mesh firstly");
		return;
	}
	else {
		pcl::io::loadPLYFile((meshPath + "\\" + fileInfo.name).c_str(), *avgMesh);
	}

	// Calculate the number of files
	int fileNumber = 0;
	for (const auto& entry : std::filesystem::directory_iterator(meshPath))
		if (std::filesystem::is_regular_file(entry) && entry.path().extension() == ".ply")
			fileNumber++;

	// Read truth value
	std::string truthFilePath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\Truth.csv";
	if (_access(truthFilePath.c_str(), 0) != 0) {
		QMessageBox::warning(nullptr, "Not file exist", "Truth value file is not exist");
		return;
	}

	std::vector<std::string> remainFile;
	do {
		std::string number = fileInfo.name;
		remainFile.emplace_back(number);
	} while (_findnext64(fileHandel, &fileInfo) == 0);

	std::vector<std::string> truthFileName;
	std::vector<double> truthValue;
	std::ifstream file(truthFilePath);
	if (!file.is_open()) {
		QMessageBox::warning(nullptr, "File open error", "Can't open file Truth.csv");
		return;
	}
	std::string line;
	std::getline(file, line);//get the title row
	while (std::getline(file, line)) {
		std::string cell;
		size_t pos = 0;
		if ((pos = line.find(',')) != std::string::npos) {
			cell = line.substr(0, pos);
			line.erase(0, pos + 1);

			std::string oldSubstring = "depth";
			std::string newSubstring = "mesh";

			size_t pos2 = 0;
			while ((pos2 = cell.find(oldSubstring, pos2)) != std::string::npos) {
				cell.replace(pos2, oldSubstring.length(), newSubstring);
				pos2 += newSubstring.length();
			}
			truthFileName.emplace_back(cell);
		}
		truthValue.emplace_back(std::stod(line));
	}
	file.close();

	for (int i = 0; i < truthFileName.size(); i++) {
		std::string oldStr = "png";
		std::string newStr = "ply";
		size_t pos = 0;
		if ((pos = truthFileName[i].find(oldStr, pos)) != std::string::npos)
			truthFileName[i].replace(pos, oldStr.length(), newStr);
	}

	//Remove parameters corresponding to files that failed reconstruction
	for (int i = truthFileName.size() - 1; i >= 0; i--) {
		auto it = std::find(remainFile.begin(), remainFile.end(), truthFileName[i]);
		if (it == remainFile.end()) {
			truthFileName.erase(truthFileName.begin() + i);
			truthValue.erase(truthValue.begin() + i);
		}
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::fromPCLPointCloud2(avgMesh->cloud, *cloud);
	std::vector<pcl::Vertices> polygon = avgMesh->polygons;
	Eigen::MatrixX3d V_0(cloud->size(), 3);
	Eigen::MatrixX3i T_0(polygon.size(), 3);
	for (int j = 0; j < cloud->size(); j++) {
		V_0(j, 0) = cloud->points[j].x;
		V_0(j, 1) = cloud->points[j].y;
		V_0(j, 2) = cloud->points[j].z;
	}
	for (int j = 0; j < polygon.size(); j++) {
		T_0(j, 0) = polygon[j].vertices.at(0);
		T_0(j, 1) = polygon[j].vertices.at(1);
		T_0(j, 2) = polygon[j].vertices.at(2);
	}

	//Translate the target point's center of gravity to the origin
	Eigen::VectorXd centroid = V_0.colwise().mean();
	V_0.rowwise() -= centroid.transpose();
	//normalization
	V_0 /= V_0.norm();

	int verticesNum = V_0.rows();
	std::vector<Eigen::MatrixX3d> V;

	//Translate the source points' center of gravity to the origin and normalization it
	for (int j = 0; j < fileNumber; j++) {
		pcl::PolygonMesh::Ptr trainMesh(new pcl::PolygonMesh);
		pcl::io::loadPLYFile(meshPath + "\\" + truthFileName[j], *trainMesh);
		pcl::PointCloud<pcl::PointXYZ>::Ptr trainPc(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::fromPCLPointCloud2(trainMesh->cloud, *trainPc);
		Eigen::MatrixX3d trainVertices(trainPc->size(), 3);
		for (int k = 0; k < trainPc->size(); k++) {
			trainVertices(k, 0) = trainPc->points[k].x;
			trainVertices(k, 1) = trainPc->points[k].y;
			trainVertices(k, 2) = trainPc->points[k].z;
		}
		Eigen::VectorXd centroidVertices = trainVertices.colwise().mean();
		trainVertices.rowwise() -= centroidVertices.transpose();
		trainVertices /= trainVertices.norm();
		V.emplace_back(trainVertices);
	}

	//Alignment
	int iter = 0;
	while (iter < 10) {
		iter++;

		//Align all shapes to the average shape
		for (int j = 0; j < fileNumber; j++)
			Procrustes(V_0, V[j]);

		Eigen::MatrixX3d V_0_temp = GenerateAvg(V, fileNumber, verticesNum);
		Procrustes(V_0, V_0_temp);
		double loss = (V_0_temp - V_0).norm();
		V_0 = V_0_temp;

		QTextEdit* outputInfo = ui.outputInfoText;
		outputInfo->append(("Current iterations is : " + std::to_string(iter) + ", loss is: " + std::to_string(loss)).c_str());
		QApplication::processEvents(); //Update UI response
	}

	Eigen::VectorXd V_0Centroid = V_0.colwise().mean();
	Eigen::MatrixXd centered = V_0.rowwise() - V_0Centroid.transpose();
	// Calculate Covariance matrix
	Eigen::MatrixXd covarianceMatrix = (centered.adjoint() * centered) / double(centered.rows() - 1);

	// Calculate feature vectors
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covarianceMatrix);
	Eigen::MatrixXcd eigenVectorsComplex = eigenSolver.eigenvectors(); // Complex eigenvector
	Eigen::VectorXd eigenValues = eigenSolver.eigenvalues().real();

	std::vector<int> sortedIndices(eigenValues.size());
	for (int i = 0; i < eigenValues.size(); ++i)
		sortedIndices[i] = i;

	std::sort(sortedIndices.begin(), sortedIndices.end(), [&](int a, int b) {
		return eigenValues(a) > eigenValues(b);
		});

	Eigen::MatrixXd eigenVectors(3, 3);
	for (int i = 0; i < 3; i++)
		eigenVectors.col(i) = eigenVectorsComplex.real().col(sortedIndices[i]);

	//Extract the real part as the principal component matrix
	Eigen::Matrix3d rotateX;
	rotateX << 1, 0, 0,
		0, -1, 0,
		0, 0, -1;
	Eigen::VectorXd centroidNorm = V_0.colwise().mean();
	V_0 = (V_0.rowwise() - centroidNorm.transpose()) * eigenVectors * rotateX;
	for (int j = 0; j < fileNumber; j++)
		V[j] = (V[j].rowwise() - centroidNorm.transpose()) * eigenVectors * rotateX;

	auto minValue = std::min_element(truthValue.begin(), truthValue.end());
	auto maxValue = std::max_element(truthValue.begin(), truthValue.end());
	double minTruth = *minValue;
	double maxTruth = *maxValue;

	//Calculate scores in sets
	Eigen::VectorXd score(fileNumber);
	for (int j = 0; j < fileNumber; j++)
		score[j] = EvaluateScore(V_0, V[j], T_0);

	// Write the mesh after align
	if (ui.writeAlignCheckBox->isChecked()) {
		std::string alignPath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\alignmesh";
		if (_access(alignPath.c_str(), 0) == 0)
			DeleteDiretoryContents(alignPath);//clear the folder

		for (int i = 0; i < fileNumber; i++) {
			if (_access(alignPath.c_str(), 0) != 0)
				if (_mkdir(alignPath.c_str()) == -1) {
					QMessageBox::warning(nullptr, "mkdir error", "mkdir error");
					return;
				}

			pcl::PointCloud<pcl::PointXYZ>::Ptr alignCloud(new pcl::PointCloud<pcl::PointXYZ>);
			alignCloud->resize(V_0.rows());
			for (int j = 0; j < V_0.rows(); j++) {
				alignCloud->at(j).x = V[i](j, 0);
				alignCloud->at(j).y = V[i](j, 1);
				alignCloud->at(j).z = V[i](j, 2);
			}
			pcl::PolygonMesh::Ptr alignMesh(new pcl::PolygonMesh);
			pcl::toPCLPointCloud2(*alignCloud, alignMesh->cloud);
			alignMesh->polygons.resize(T_0.rows());
			for (int j = 0; j < T_0.rows(); j++) {
				pcl::Vertices v1;
				v1.vertices.resize(3);
				v1.vertices[0] = T_0(j, 0);
				v1.vertices[1] = T_0(j, 1);
				v1.vertices[2] = T_0(j, 2);
				alignMesh->polygons[j] = v1;
			}
			pcl::io::savePLYFile((alignPath + "\\" + truthFileName[i]).c_str(), *alignMesh);
		}
	}

	double minScore = score.minCoeff();
	double maxScore = score.maxCoeff();
	pcl::PointCloud<pcl::PointXYZ>::Ptr testPc(new pcl::PointCloud<pcl::PointXYZ>);
	testPc->resize(V_0.rows());
	for (int i = 0; i < V_0.rows(); i++) {
		testPc->at(i).x = V_0(i, 0);
		testPc->at(i).y = V_0(i, 1);
		testPc->at(i).z = V_0(i, 2);
	}
	pcl::PolygonMesh::Ptr modelMesh(new pcl::PolygonMesh);
	pcl::toPCLPointCloud2(*testPc, modelMesh->cloud);
	modelMesh->polygons.resize(T_0.rows());
	for (int i = 0; i < T_0.rows(); i++) {
		pcl::Vertices verticesModel;
		verticesModel.vertices.resize(3);
		verticesModel.vertices[0] = T_0(i, 0);
		verticesModel.vertices[1] = T_0(i, 1);
		verticesModel.vertices[2] = T_0(i, 2);
		modelMesh->polygons[i] = verticesModel;
	}
	pcl::io::savePLYFile((std::string(mainPath) + "\\" + _projectName.toStdString() + "\\model.ply").c_str(), *modelMesh);
	std::ofstream modelParaFile(std::string(mainPath) + "\\" + _projectName.toStdString() + "\\model.csv");
	if (!modelParaFile.is_open()) {
		QMessageBox::warning(nullptr, "Can't open file", "Can't open file");
		return;
	}
	modelParaFile << std::to_string(minTruth) + "," + std::to_string(maxTruth) + "," + std::to_string(minScore) + "," +
		std::to_string(maxScore) << std::endl;
	modelParaFile.close();
	QTextEdit* outputInfo = ui.outputInfoText;
	outputInfo->append("Training complete");
}


void LShapeAnalyser_single::browseDLC() {
	QString filePath = QFileDialog::getExistingDirectory(this, "Select DLC project folder", "", QFileDialog::ShowDirsOnly);
	if (!filePath.isEmpty()) {
		QLineEdit* DLCLine = ui.DLCPathLineEdit;
		DLCLine->setText(filePath);
	}
}


void LShapeAnalyser_single::editPara() {
	QTabWidget* tab = ui.tabWidget;
	tab->setCurrentWidget(ui.convertToMeshTab);
}


void LShapeAnalyser_single::score() {
	// import file
	QString folderPath = QFileDialog::getExistingDirectory(this, "Select depth image folder", "", QFileDialog::ShowDirsOnly);
	if (folderPath.isEmpty()) 
		return;
	
	char mainPath[_MAX_PATH];
	if (!_getcwd(mainPath, _MAX_PATH)) {
		QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
		return;
	}
	std::string scorePath = folderPath.toStdString() + "\\score_temp";
	if (_access(scorePath.c_str(), 0) == 0) {
		DeleteDiretoryContents(scorePath);
		std::filesystem::remove(scorePath);
	}
	if (_mkdir(scorePath.c_str()) != 0) { //create a temporary folder to store intermediate file
		QMessageBox::warning(nullptr, "Folder gray create failure", "Folder gray create failure");
		return;
	}
	QTextEdit* outputInfo = ui.outputInfoText;

	// Calculate the number of files
	int numFiles = 0;
	for (const auto& entry : std::filesystem::directory_iterator(folderPath.toStdString())) {
		if (std::filesystem::is_regular_file(entry)) {
			numFiles++;
		}
	}
	int quantile = floor(log10(numFiles) + 1); //Quantile of document

	std::vector<std::string> truthFileName;
	_finddatai64_t fileInfo;
	intptr_t fileHandel = 0;
	if ((fileHandel = _findfirst64((folderPath.toStdString() + "\\*.png").c_str(), &fileInfo)) != -1) {
		int number = 0;
		cv::VideoWriter writer;
		QProgressDialog progressDiglog;
		progressDiglog.setLabelText("Importing...");
		progressDiglog.setRange(0, 100);
		Qt::WindowFlags flags = progressDiglog.windowFlags();
		progressDiglog.setWindowFlags(flags | Qt::WindowStaysOnTopHint);
		do {
			progressDiglog.setValue(number * 100 / numFiles);
			QApplication::processEvents(); //Update UI response
			number++;
			truthFileName.emplace_back(fileInfo.name);
			std::string sourceFile = folderPath.toStdString() + "\\" + fileInfo.name;

			// Write the gray image and the corresponding video with extension .avi
			cv::Mat depthImage = cv::imread(sourceFile, cv::IMREAD_UNCHANGED);
			if (depthImage.empty()) {
				QMessageBox::warning(nullptr, "Can't read the depth image", "Can't read the depth image");
				return;
			}
			double depthMin, depthMax;
			cv::minMaxLoc(depthImage, &depthMin, &depthMax);
			int h = depthImage.cols;
			int w = depthImage.rows;
			cv::Mat grayImage = cv::Mat::zeros(depthImage.size(), CV_8UC1);
			//Grayscale pixels
			for (int col = 0; col < h; col++)
				for (int row = 0; row < w; row++)
					grayImage.ptr(row)[col] = (depthImage.ptr(row)[col] - depthMin) / (depthMax - depthMin) * 255.0;

			std::stringstream idxFill;
			idxFill << std::setw(quantile) << std::setfill('0') << number; //Fill in '0' before numbers
			std::string grayTargetFile = scorePath + "\\gray_" + idxFill.str() + ".png";
			cv::imwrite(grayTargetFile, grayImage);
			if (number == 1)
				writer = cv::VideoWriter(scorePath + "\\Synthetic.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 32, cv::Size(h, w), true);

			cv::Mat image = cv::imread(grayTargetFile);
			writer << image;
		} while (_findnext64(fileHandel, &fileInfo) == 0);
		progressDiglog.close();
		writer.release();
		outputInfo->append("Import depth file from " + folderPath + ", and the gray image and synthetic video have been created.");
	}
	else {//fileHandel==-1
		QMessageBox::warning(nullptr, "Not file in this folder", "Not file in this folder");
		return;
	}


	//DLC
	QLineEdit* DLCPath = ui.DLCPathLineEdit;
	std::string videoFilePath = scorePath + "\\Synthetic.avi";
	outputInfo->append("Opening DLC");
	QApplication::processEvents(); //Update UI response
	std::ofstream pyFile(scorePath + "\\DLC_analyze.py");
	if (!pyFile.is_open()) {
		QMessageBox::warning(nullptr, "file create fail", "file create fail");
		return;
	}
	std::string code1 = "import deeplabcut";
	std::string code2 = "config_path=r\"" + DLCPath->text().toStdString() + "\\config.yaml\"";
	std::string code3 = "videoFilePath=r\"" + videoFilePath + "\"";
	std::string code4 = "deeplabcut.analyze_videos(config_path,[videoFilePath],save_as_csv=True)";
	pyFile << code1 << std::endl << code2 << std::endl << code3 << std::endl << code4 << std::endl;
	pyFile.close();
	std::string pyCommand = "conda activate DEEPLABCUT && python \"" + scorePath + "\\DLC_analyze.py\"";
	size_t found = pyCommand.find("/");
	while (found != std::string::npos) {
		pyCommand.replace(found, 1, "\\");
		found = pyCommand.find("/", found + 1);
	}
	std::system(pyCommand.c_str());
	outputInfo->append("Keypoint detect finish");
	QApplication::processEvents(); //Update UI response


	// convert to mesh
	pcl::PolygonMesh::Ptr modelMesh(new pcl::PolygonMesh);
	pcl::io::loadPLYFile((std::string(mainPath) + "\\" + _projectName.toStdString() + "\\model.ply").c_str(), *modelMesh);
	double fx = ui.fxLineEdit->text().toDouble();
	double fy = ui.fyLineEdit->text().toDouble();
	double cx = ui.cxLineEdit->text().toDouble();
	double cy = ui.cyLineEdit->text().toDouble();

	//Extract the predicted marker point coordinates
	_finddatai64_t fileInfoConvert;
	intptr_t fileHandelConvert = 0;
	if ((fileHandelConvert = _findfirst64((scorePath + "\\*.csv").c_str(), &fileInfoConvert)) == -1) {
		QMessageBox::warning(nullptr, "No key point infomation", "Key points have not been detected yet.");
		return;
	}

	std::string csvFilePath = scorePath + "\\" + fileInfoConvert.name;

	//read keyPoint coordinate from table element 
	std::vector<std::vector<double>> markPoint;
	std::vector<std::vector<double>> likelihood;
	std::string lineStr;
	std::ifstream csvFile(csvFilePath);
	if (!csvFile.is_open()) {
		QMessageBox::warning(nullptr, "File not found", "File not found");
		return;
	}
	int notPointCount = 0;
	while (std::getline(csvFile, lineStr)) {
		if (notPointCount < 3) {//The first three lines of the file are attribute names
			notPointCount++;
			continue;
		}
		int count = 0;
		std::vector<double> pointRow;
		std::vector<double> likeRow;
		std::stringstream ss(lineStr);

		std::string cell;
		while (std::getline(ss, cell, ',')) {
			if (count != 0) {// the first col is number
				double value = std::stod(cell);
				if (count % 3 == 0)
					likeRow.emplace_back(value);
				else
					pointRow.emplace_back(value);
			}
			count++;
		}
		markPoint.emplace_back(pointRow);
		likelihood.emplace_back(likeRow);
	}
	csvFile.close();
	int markNum = markPoint[0].size() / 2;
	outputInfo->append("Start convert to mesh.");

	Eigen::MatrixX3d verticesTarget;
	Eigen::MatrixX3i facesTarget;
	Eigen::VectorXi boundaryTarget;
	Eigen::VectorXi selectConesTarget;
	verticesTarget.resize(modelMesh->cloud.width, 3);
	facesTarget.resize(modelMesh->polygons.size(), 3);

	pcl::PointCloud<pcl::PointXYZ>::Ptr pcTarget(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::fromPCLPointCloud2(modelMesh->cloud, *pcTarget);
	for (int i = 0; i < pcTarget->width; i++) {
		verticesTarget(i, 0) = pcTarget->points[i].x;
		verticesTarget(i, 1) = pcTarget->points[i].y;
		verticesTarget(i, 2) = pcTarget->points[i].z;
	}
	for (int i = 0; i < modelMesh->polygons.size(); i++) {
		pcl::Vertices::Ptr tempVertices(new pcl::Vertices);
		*tempVertices = modelMesh->polygons[i];
		facesTarget(i, 0) = (*tempVertices).vertices[0];
		facesTarget(i, 1) = (*tempVertices).vertices[1];
		facesTarget(i, 2) = (*tempVertices).vertices[2];
	}

	Surface_mesh cgalMesh;
	cgalMesh.clear();
	for (const auto& point : pcTarget->points)
		cgalMesh.add_vertex(Point_3(point.x, point.y, point.z));
	for (const auto& polygon : modelMesh->polygons)
		cgalMesh.add_face(Vertex_index(polygon.vertices[0]), Vertex_index(polygon.vertices[1]), Vertex_index(polygon.vertices[2]));
	Surface_mesh::Halfedge_range halfEdges = cgalMesh.halfedges();
	//get the unorder free boundary
	for (auto it = halfEdges.begin(); it != halfEdges.end(); it++)
		if (cgalMesh.is_border(*it)) {
			boundaryTarget.conservativeResize(boundaryTarget.size() + 1);
			Vertex_index boundaryIndex = cgalMesh.source(*it);
			boundaryTarget[boundaryTarget.size() - 1] = boundaryIndex;
		}

	std::unordered_map<Vertex_index, Vertex_index> nextBoundaryVertex;
	for (auto it = halfEdges.begin(); it != halfEdges.end(); it++)
		if (cgalMesh.is_border(*it)) {
			Vertex_index boundaryIndex = cgalMesh.source(*it);
			Vertex_index nextBoundaryIndex = cgalMesh.target(*it);
			nextBoundaryVertex[boundaryIndex] = nextBoundaryIndex;
		}

	Vertex_index boundaryTargetStart = Vertex_index(boundaryTarget[0]);
	size_t index = 0;
	bool firstError = false;
	while (true) {
		boundaryTarget[index] = boundaryTargetStart;
		index++;
		auto it = nextBoundaryVertex.find(boundaryTargetStart);
		if (index == boundaryTarget.size())
			break;

		if (it == nextBoundaryVertex.end()) {
			firstError = true;
			break;
		}
		boundaryTargetStart = it->second;
		nextBoundaryVertex.erase(it);
	}

	std::ifstream modelFile(std::string(mainPath) + "/" + _projectName.toStdString() + "/modelKeyPointIndex.txt");
	std::string keyPointIdx;
	pcl::PointCloud<pcl::PointXYZ>::Ptr verticesKeyPointI(new pcl::PointCloud<pcl::PointXYZ>);
	while (std::getline(modelFile, keyPointIdx))
		verticesKeyPointI->push_back(pcTarget->points[std::stoi(keyPointIdx)]);

	modelFile.close();

	//Using the position of the first keyPoint as the starting index
	int nearestKeyPoint1Index = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(0));
	Eigen::VectorXi nearestKeyPoint1Head = boundaryTarget.head(nearestKeyPoint1Index);
	boundaryTarget.segment(0, boundaryTarget.size() - nearestKeyPoint1Index) =
		boundaryTarget.segment(nearestKeyPoint1Index, boundaryTarget.size() - nearestKeyPoint1Index);
	boundaryTarget.segment(boundaryTarget.size() - nearestKeyPoint1Index, nearestKeyPoint1Index) = nearestKeyPoint1Head;

	////search the corners around the egde
	if (verticesKeyPointI->size() == 3) {
		int nearestVertices1IdxTarget = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(0));
		int nearestVertices2IdxTarget = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(1));
		int nearestVertices3IdxTarget = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(2));
		selectConesTarget.resize(3);
		selectConesTarget[0] = nearestVertices1IdxTarget;
		selectConesTarget[1] = nearestVertices2IdxTarget;
		selectConesTarget[2] = nearestVertices3IdxTarget;
	}
	else {
		double interval = verticesKeyPointI->size() / 4.0;
		int nearestVertices1IdxTarget = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(round(interval * 1 - 1)));
		int nearestVertices2IdxTarget = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(round(interval * 2 - 1)));
		int nearestVertices3IdxTarget = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(round(interval * 3 - 1)));
		int nearestVertices4IdxTarget = Dsearchn(pcTarget, boundaryTarget, verticesKeyPointI->at(round(interval * 4 - 1)));
		selectConesTarget.resize(4);
		selectConesTarget[0] = nearestVertices1IdxTarget;
		selectConesTarget[1] = nearestVertices2IdxTarget;
		selectConesTarget[2] = nearestVertices3IdxTarget;
		selectConesTarget[3] = nearestVertices4IdxTarget;
	}
	std::sort(selectConesTarget.data(), selectConesTarget.data() + selectConesTarget.size());

	// Process each image
	std::string dataName;
	int canNotMapCount = 0;//calculate the count of mesh that can't be mapped
	int normalEstimation = -1;//Indicates that the nth file is being processed

	QProgressDialog bar;
	bar.setLabelText("Converting...");
	bar.setRange(0, 100);
	Qt::WindowFlags barFlag = bar.windowFlags();
	bar.setWindowFlags(barFlag | Qt::WindowStaysOnTopHint);
	for (const auto& entry : std::filesystem::directory_iterator(folderPath.toStdString().c_str())) {
		if (std::filesystem::is_regular_file(entry) && entry.path().extension() == ".png") {
			normalEstimation++;
			bar.setValue(normalEstimation * 100 / numFiles);
			QApplication::processEvents(); //Update UI response
			if (_stopExecution) {
				QMessageBox::information(nullptr, "Cancel", "User cancel the convertion");
				bar.setValue(0);
				_stopExecution = false;
				return;
			}

			std::stringstream indexFill;
			indexFill << std::setw(quantile) << std::setfill('0') << normalEstimation + 1; //Fill in '0' before numbers
			dataName = indexFill.str();

			cv::Mat depthImg = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
			int row = depthImg.rows;
			int col = depthImg.cols;

			std::vector<double> cornerCoordinate = markPoint[normalEstimation];
			int keyPointNum = markPoint[0].size() / 2;

			std::vector<Point_2d> keyPoint;

			bool isDepthMissing = false;
			for (int i = 0; i < keyPointNum; i++) {
				Point_2d nullPoint = { 0,0 };
				keyPoint.emplace_back(nullPoint);
				keyPoint[i].x = std::floor(cornerCoordinate[2 * static_cast<double>(i)]);
				keyPoint[i].y = std::floor(cornerCoordinate[2 * static_cast<double>(i) + 1]);
				if (depthImg.at<uint16_t>(keyPoint[i].y, keyPoint[i].x) == 0) {
					outputInfo->append(("The depth of the key point of file " + entry.path().filename().string() + " is missing.").c_str());
					canNotMapCount++;
					isDepthMissing = true;
					break;
				}
			}
			if (isDepthMissing)
				continue;//Process the next image
			bool isKeyDevia = false;
			for (int i = 0; i < keyPointNum; i++) {
				if (likelihood[normalEstimation][i] < 0.95) {
					outputInfo->append(("The keyPoint of file " + entry.path().filename().string() + " is deviation.").c_str());
					canNotMapCount++;
					isKeyDevia = true;
					break;
				}
			}
			if (isKeyDevia)
				continue;//Process the next image


			// Convert to 3D point
			Point_2d p;
			pcl::PointCloud<pcl::PointXYZ>::Ptr originalVertices(new pcl::PointCloud<pcl::PointXYZ>);
			if (!isConcavePolygon(keyPoint)) {// the polygon enclosed by key points is a convex polygon
				for (int i = 0; i < col; i++) {
					for (int j = 0; j < row; j++) {
						p.x = i;
						p.y = j;
						if (isPointInConvexPolygon(keyPoint, p)) {
							double depth = depthImg.at<uint16_t>(j, i);

							pcl::PointXYZ point;
							point.x = (j - cx) * depth / fx;
							point.y = (i - cy) * depth / fy;
							point.z = depth;
							originalVertices->points.push_back(point);
						}
					}
				}
			}
			else {// the polygon enclosed by key points is a concave polygon
				for (int i = 0; i < col; i++) {
					for (int j = 0; j < row; j++) {
						p.x = i;
						p.y = j;
						if (isPointInConcavePolygon(keyPoint, p)) {
							double depth = depthImg.at<uint16_t>(j, i);

							pcl::PointXYZ point;
							point.x = (j - cx) * depth / fx;
							point.y = (i - cy) * depth / fy;
							point.z = depth;
							originalVertices->points.push_back(point);
						}
					}
				}
			}

			//add the keyPoint
			for (int i = 0; i < keyPointNum; i++) {
				double depth = depthImg.at<uint16_t>(keyPoint[i].y, keyPoint[i].x);

				pcl::PointXYZ point;
				point.x = (keyPoint[i].y - cx) * depth / fx;
				point.y = (keyPoint[i].x - cy) * depth / fy;
				point.z = depth;
				originalVertices->points.push_back(point);
			}
			originalVertices->width = originalVertices->points.size();
			originalVertices->height = 1;

			pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutlierRemoved(new pcl::PointCloud<pcl::PointXYZ>);
			originalVertices = TranslateAndNormalizePointCloud(originalVertices);
			pcl::StatisticalOutlierRemoval<pcl::PointXYZ> statisticalOutlierRemoval;
			statisticalOutlierRemoval.setInputCloud(originalVertices);
			statisticalOutlierRemoval.setMeanK(50);
			statisticalOutlierRemoval.setStddevMulThresh(1.0);
			statisticalOutlierRemoval.filter(*cloudOutlierRemoved);

			pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDownsampled(new pcl::PointCloud<pcl::PointXYZ>);
			int downLevel = ui.downLvSpinBox->value();
			if (downLevel != 0) {
				pcl::VoxelGrid<pcl::PointXYZ> downSampled;
				downSampled.setInputCloud(cloudOutlierRemoved);
				double leaf = 0.0001 * pow(2, (downLevel - 1));
				downSampled.setLeafSize(leaf, leaf, leaf);
				downSampled.filter(*cloudDownsampled);
			}

			//Reconstruct mesh
			pcl::PointCloud<pcl::PointNormal>::Ptr cloudWithNormal(new pcl::PointCloud<pcl::PointNormal>);
			pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
			pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>);
			pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

			tree->setInputCloud(cloudDownsampled);
			normalEstimation.setInputCloud(cloudDownsampled);
			normalEstimation.setSearchMethod(tree);
			normalEstimation.setKSearch(20);
			normalEstimation.setViewPoint(0, 0, DBL_MAX);//Set viewpoint at infinity
			normalEstimation.compute(*normal);
			pcl::concatenateFields(*cloudDownsampled, *normal, *cloudWithNormal);

			pcl::Poisson<pcl::PointNormal>  poisson;
			poisson.setInputCloud(cloudWithNormal);
			pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
			poisson.reconstruct(*mesh);

			pcl::PointCloud<pcl::PointXYZ>::Ptr verticesOriginal(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::fromPCLPointCloud2(mesh->cloud, *verticesOriginal);
			std::vector<pcl::Vertices> facesOriginal;
			facesOriginal.assign(mesh->polygons.begin(), mesh->polygons.end());
			std::vector<double> verticesX, verticesY, verticesZ;
			for (const pcl::PointXYZ& point : verticesOriginal->points) {
				verticesX.emplace_back(point.x);
				verticesY.emplace_back(point.y);
				verticesZ.emplace_back(point.z);
			}

			//extract keypoints in 3D space
			pcl::PointCloud<pcl::PointXYZ>::Ptr verticesKeyPointI(new pcl::PointCloud<pcl::PointXYZ>);
			std::vector<double> verticesXKeyPoint, verticesYKeyPoint, verticesZKeyPoint;
			for (int i = 0; i < keyPointNum; i++) {
				pcl::PointXYZ point;
				point.x = originalVertices->at(originalVertices->size() - keyPointNum + i).x;
				point.y = originalVertices->at(originalVertices->size() - keyPointNum + i).y;
				point.z = originalVertices->at(originalVertices->size() - keyPointNum + i).z;
				verticesXKeyPoint.emplace_back(point.x);
				verticesYKeyPoint.emplace_back(point.y);
				verticesZKeyPoint.emplace_back(point.z);
				verticesKeyPointI->points.push_back(point);
			}

			Eigen::Map<Eigen::VectorXd> fitVecticesX(verticesXKeyPoint.data(), verticesXKeyPoint.size());
			Eigen::Map<Eigen::VectorXd> fitVecticesY(verticesYKeyPoint.data(), verticesYKeyPoint.size());
			Eigen::Map<Eigen::VectorXd> fitVecticesZ(verticesZKeyPoint.data(), verticesZKeyPoint.size());

			//Fit a plane with key points
			Eigen::MatrixXd fitPlaneCoefficient(keyPointNum, 3);
			fitPlaneCoefficient << fitVecticesX, fitVecticesY, Eigen::VectorXd::Ones(keyPointNum);

			////Using the Least Squares Method to Solve the Coefficients of a Plane Equation
			Eigen::VectorXd fittedPlaneCoefficient = fitPlaneCoefficient.colPivHouseholderQr().solve(fitVecticesZ);
			std::vector<int> preserveVerticesIndex;
			if (!isConcavePolygon(keyPoint)) {// the polygon enclosed by key points is a convex polygon
				double N1 = fittedPlaneCoefficient(0);
				double N2 = fittedPlaneCoefficient(1);
				double N3 = -1.0;
				Eigen::VectorXd A(keyPointNum);
				Eigen::VectorXd B(keyPointNum);
				Eigen::VectorXd C(keyPointNum);
				Eigen::VectorXd depthImg(keyPointNum);

				//Solve the coefficients of planes
				for (int i = 0; i < keyPointNum; i++) {
					Eigen::Vector4d ABCD;
					ABCD = SolveSurface(verticesKeyPointI->at(i), verticesKeyPointI->at((i + 1) % keyPointNum), N1, N2, N3);
					A[i] = ABCD[0];
					B[i] = ABCD[1];
					C[i] = ABCD[2];
					depthImg[i] = ABCD[3];
				}

				std::vector<std::vector<int>> preserveVerticesIndexI;
				preserveVerticesIndexI.resize(keyPointNum);

				// Find the interior point where three planes intersect
				for (int i = 0; i < keyPointNum; i++) {
					//Using key points that are not on the plane as discriminant points
					if (A((i + 1) % keyPointNum) * verticesKeyPointI->at(i).x + B((i + 1) % keyPointNum) *
						verticesKeyPointI->at(i).y + C((i + 1) % keyPointNum) * verticesKeyPointI->at(i).z +
						depthImg((i + 1) % keyPointNum) <= 0) {
						for (int j = 0; j < verticesOriginal->size(); j++)
							if (A((i + 1) % keyPointNum) * verticesOriginal->at(j).x + B((i + 1) % keyPointNum) *
								verticesOriginal->at(j).y + C((i + 1) % keyPointNum) * verticesOriginal->at(j).z +
								depthImg((i + 1) % keyPointNum) <= 0)
								preserveVerticesIndexI[i].emplace_back(j);
					}
					else {
						for (int j = 0; j < verticesOriginal->size(); j++)
							if (A((i + 1) % keyPointNum) * verticesOriginal->at(j).x + B((i + 1) % keyPointNum) *
								verticesOriginal->at(j).y + C((i + 1) % keyPointNum) * verticesOriginal->at(j).z +
								depthImg((i + 1) % keyPointNum) >= 0)
								preserveVerticesIndexI[i].emplace_back(j);
					}
					if (i == 0)
						preserveVerticesIndex = preserveVerticesIndexI[i];
					else {
						std::sort(preserveVerticesIndexI[i].begin(), preserveVerticesIndexI[i].end());
						std::sort(preserveVerticesIndex.begin(), preserveVerticesIndex.end());
						auto it = std::set_intersection(preserveVerticesIndexI[i].begin(), preserveVerticesIndexI[i].end(),
							preserveVerticesIndex.begin(), preserveVerticesIndex.end(), preserveVerticesIndex.begin());
						preserveVerticesIndex.resize(it - preserveVerticesIndex.begin());
					}
				}
			}
			else {// the polygon enclosed by key points is a concave polygon
				// Map the point set to the fitted plane,Ax+By+Cz+depthImg=0 A=fittedPlaneCoefficient(1),
				// B=fittedPlaneCoefficient(2),C=-1,depthImg=fittedPlaneCoefficient(3)
				Eigen::VectorXd verticesXMap, verticesYMap, verticesZMap;
				for (int i = 0; i < verticesX.size(); i++) {
					verticesXMap[i] = (verticesX[i] - fittedPlaneCoefficient[0] * (fittedPlaneCoefficient[0] * verticesX[i] +
						fittedPlaneCoefficient[1] * verticesY[i] - verticesZ[i] + fittedPlaneCoefficient[2]) /
						(pow(fittedPlaneCoefficient[0], 2) + pow(fittedPlaneCoefficient[1], 2) + 1));
					verticesYMap[i] = (verticesY[i] - fittedPlaneCoefficient[1] * (fittedPlaneCoefficient[0] * verticesX[i] +
						fittedPlaneCoefficient[1] * verticesY[i] - verticesZ[i] + fittedPlaneCoefficient[2]) /
						(pow(fittedPlaneCoefficient[0], 2) + pow(fittedPlaneCoefficient[1], 2) + 1));
					verticesZMap[i] = (verticesZ[i] + (fittedPlaneCoefficient[0] * verticesX[i] + fittedPlaneCoefficient[1] *
						verticesY[i] - verticesZ[i] + fittedPlaneCoefficient[2]) / (pow(fittedPlaneCoefficient[0], 2) +
							pow(fittedPlaneCoefficient[1], 2) + 1));
				}
				Eigen::VectorXd verticesXKeyPointMap, verticesYKeyPointMap, verticesZKeyPointMap;
				for (int i = 0; i < verticesXKeyPoint.size(); i++) {
					verticesXKeyPointMap[i] = (verticesXKeyPoint[i] - fittedPlaneCoefficient[0] * (fittedPlaneCoefficient[0] *
						verticesXKeyPoint[i] + fittedPlaneCoefficient[1] * verticesYKeyPoint[i] - verticesZKeyPoint[i] +
						fittedPlaneCoefficient[2]) / (pow(fittedPlaneCoefficient[0], 2) + pow(fittedPlaneCoefficient[1], 2) + 1));
					verticesYKeyPointMap[i] = (verticesYKeyPoint[i] - fittedPlaneCoefficient[1] * (fittedPlaneCoefficient[0] *
						verticesXKeyPoint[i] + fittedPlaneCoefficient[1] * verticesYKeyPoint[i] - verticesZKeyPoint[i] +
						fittedPlaneCoefficient[2]) / (pow(fittedPlaneCoefficient[0], 2) + pow(fittedPlaneCoefficient[1], 2) + 1));
					verticesZKeyPointMap[i] = (verticesZKeyPoint[i] + (fittedPlaneCoefficient[0] * verticesXKeyPoint[i] +
						fittedPlaneCoefficient[1] * verticesYKeyPoint[i] - verticesZKeyPoint[i] + fittedPlaneCoefficient[2]) /
						(pow(fittedPlaneCoefficient[0], 2) + pow(fittedPlaneCoefficient[1], 2) + 1));
				}

				//Rotate point to xOy plane
				Eigen::Vector3d fitPlaneNormal = { fittedPlaneCoefficient[0],fittedPlaneCoefficient[1],-1 };
				Eigen::Vector3d xOy_n = { 0,0,1 };
				double dot = fitPlaneNormal.dot(xOy_n);
				double normFit = fitPlaneNormal.norm();
				double normxOy = xOy_n.norm();
				double theta = std::acos(dot / (normFit * normxOy));
				Eigen::Vector3d rotateAxis = fitPlaneNormal.cross(xOy_n) / (fitPlaneNormal.cross(xOy_n)).norm();
				Eigen::MatrixXd verticesMap;
				verticesMap << verticesXMap, verticesYMap, verticesZMap;
				Eigen::MatrixXd verticesKeyPointMap;
				verticesKeyPointMap << verticesXKeyPointMap, verticesYKeyPointMap, verticesZKeyPointMap;
				Eigen::MatrixXd verticesRotated = RotatePoint(verticesMap, theta, rotateAxis);
				Eigen::MatrixXd verticesKeyPointRotated = RotatePoint(verticesKeyPointMap, theta, rotateAxis);

				//Map the rotate point to 2D space
				std::vector<Point_2d> verticesXYKeyPointRotated;
				for (int i = 0; i < keyPointNum; i++) {
					Point_2d p;
					p.x = verticesKeyPointRotated(i, 0);
					p.y = verticesKeyPointRotated(i, 1);
					verticesXYKeyPointRotated.emplace_back(p);
				}
				for (int i = 0; i < verticesX.size(); i++) {
					Point_2d p;
					p.x = verticesRotated(i, 0);
					p.y = verticesRotated(i, 1);
					if (isPointInConcavePolygon(verticesXYKeyPointRotated, p))
						preserveVerticesIndex.emplace_back(i);
				}
			}
			RemoveVerticesAndFacesByValue(mesh, preserveVerticesIndex, facesOriginal);

			Eigen::MatrixX3d verticesSource(mesh->cloud.width, 3);
			Eigen::MatrixX3i facesSource(mesh->polygons.size(), 3);
			Eigen::VectorXi boundarySource;
			Eigen::VectorXi selectConesSource;

			pcl::PointCloud<pcl::PointXYZ>::Ptr pcSource(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::fromPCLPointCloud2(mesh->cloud, *pcSource);

			for (int i = 0; i < pcSource->width; i++) {
				verticesSource(i, 0) = pcSource->points[i].x;
				verticesSource(i, 1) = pcSource->points[i].y;
				verticesSource(i, 2) = pcSource->points[i].z;
			}

			for (int i = 0; i < mesh->polygons.size(); i++) {
				pcl::Vertices::Ptr tempVertices(new pcl::Vertices);
				*tempVertices = mesh->polygons[i];
				facesSource(i, 0) = (*tempVertices).vertices[0];
				facesSource(i, 1) = (*tempVertices).vertices[1];
				facesSource(i, 2) = (*tempVertices).vertices[2];
			}

			Surface_mesh cgalMesh;
			cgalMesh.clear();
			for (const auto& point : pcSource->points)
				cgalMesh.add_vertex(Point_3(point.x, point.y, point.z));
			for (const auto& polygon : mesh->polygons)
				cgalMesh.add_face(Vertex_index(polygon.vertices[0]), Vertex_index(polygon.vertices[1]),
					Vertex_index(polygon.vertices[2]));
			Surface_mesh::Halfedge_range halfEdges = cgalMesh.halfedges();

			// get the free boundary
			for (auto it = halfEdges.begin(); it != halfEdges.end(); it++)
				if (cgalMesh.is_border(*it)) {
					boundarySource.conservativeResize(boundarySource.size() + 1);
					Vertex_index boundaryIndex = cgalMesh.source(*it);
					boundarySource[boundarySource.size() - 1] = boundaryIndex;
				}

			std::unordered_map<Vertex_index, Vertex_index> nextBoundaryVertex;
			for (auto it = halfEdges.begin(); it != halfEdges.end(); it++)
				if (cgalMesh.is_border(*it)) {
					Vertex_index boundaryIndex = cgalMesh.source(*it);
					Vertex_index nextBoundaryIndex = cgalMesh.target(*it);
					nextBoundaryVertex[boundaryIndex] = nextBoundaryIndex;
				}

			Vertex_index boundarySourceStart = Vertex_index(boundarySource[0]);
			size_t index = 0;
			while (true) {
				boundarySource[index] = boundarySourceStart;
				index++;
				auto it = nextBoundaryVertex.find(boundarySourceStart);
				if (index == boundarySource.size() || it == nextBoundaryVertex.end())
					break;

				boundarySourceStart = it->second;
				nextBoundaryVertex.erase(it);
			}

			//Using the position of the first keyPoint as the starting index
			int nearestKeyPoint1Index = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(0));
			Eigen::VectorXi nearestKeyPoint1Head = boundarySource.head(nearestKeyPoint1Index);
			boundarySource.segment(0, boundarySource.size() - nearestKeyPoint1Index) =
				boundarySource.segment(nearestKeyPoint1Index, boundarySource.size() - nearestKeyPoint1Index);
			boundarySource.segment(boundarySource.size() - nearestKeyPoint1Index, nearestKeyPoint1Index) = nearestKeyPoint1Head;

			Eigen::VectorXi modelKeyPointIndex(keyPointNum);
			for (int i = 0; i < keyPointNum; i++)
				modelKeyPointIndex[i] = boundarySource[Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(i))];

			//search the corners around the egde
			if (keyPointNum == 3) {
				int nearestVertices1IdxSource = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(0));
				int nearestVertices2IdxSource = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(1));
				int nearestVertices3IdxSource = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(2));
				selectConesSource.resize(3);
				selectConesSource[0] = nearestVertices1IdxSource;
				selectConesSource[1] = nearestVertices2IdxSource;
				selectConesSource[2] = nearestVertices3IdxSource;
			}
			else {
				double interval = keyPointNum / 4.0;
				int nearestVertices1IdxSource = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(round(interval * 1 - 1)));
				int nearestVertices2IdxSource = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(round(interval * 2 - 1)));
				int nearestVertices3IdxSource = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(round(interval * 3 - 1)));
				int nearestVertices4IdxSource = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(round(interval * 4 - 1)));
				selectConesSource.resize(4);
				selectConesSource[0] = nearestVertices1IdxSource;
				selectConesSource[1] = nearestVertices2IdxSource;
				selectConesSource[2] = nearestVertices3IdxSource;
				selectConesSource[3] = nearestVertices4IdxSource;
			}
			std::sort(selectConesSource.data(), selectConesSource.data() + selectConesSource.size());

			Eigen::MatrixX3d verticesMapped(verticesTarget.rows(), verticesTarget.cols());
			OrbifoldTutteEmbedding orbifold;
			if (!orbifold.disk_map(verticesTarget, facesTarget, boundaryTarget, verticesSource, facesSource, boundarySource,
				selectConesTarget, selectConesSource, verticesMapped)) {
				outputInfo->append((dataName + " can not be map.").c_str());
				canNotMapCount++;
				continue;
			}

			pcl::PointCloud<pcl::PointXYZ>::Ptr pcMapped(new pcl::PointCloud<pcl::PointXYZ>);
			pcMapped->resize(verticesMapped.rows());
			for (int i = 0; i < verticesMapped.rows(); i++) {
				pcl::PointXYZ mapPoint;
				mapPoint.x = verticesMapped(i, 0);
				mapPoint.y = verticesMapped(i, 1);
				mapPoint.z = verticesMapped(i, 2);
				pcMapped->points[i] = mapPoint;
			}

			pcl::PolygonMesh::Ptr meshMapped(new pcl::PolygonMesh);
			meshMapped->polygons.resize(facesTarget.rows());
			for (int i = 0; i < facesTarget.rows(); i++) {
				pcl::Vertices mapVertices;
				mapVertices.vertices.push_back(facesTarget(i, 0));
				mapVertices.vertices.push_back(facesTarget(i, 1));
				mapVertices.vertices.push_back(facesTarget(i, 2));
				meshMapped->polygons[i] = mapVertices;
			}
			pcl::toPCLPointCloud2(*pcMapped, meshMapped->cloud);

			std::string meshName = scorePath + "\\mesh_" + dataName + ".ply";
			pcl::io::savePLYFile(meshName, *meshMapped);
		}
	}
	bar.close();
	outputInfo->append("Convert complete.");
	outputInfo->append(("Total " + std::to_string(canNotMapCount) + " mesh reconstruction fail").c_str());


	//score
	outputInfo->append("Starting scoring...");
	//Read model parametersand load model shape
	std::string modelCsvPath = std::string(mainPath) + "\\" + _projectName.toStdString() + "\\model.csv";
	std::ifstream modelCsvFile(modelCsvPath);
	std::string modelLine;
	std::getline(modelCsvFile, modelLine);
	modelCsvFile.close();

	std::vector<std::string> modTokens;
	std::stringstream modSS(modelLine);
	for (std::string modToken; getline(modSS, modToken, ',');)
		modTokens.emplace_back(modToken);

	double minTruth = std::stod(modTokens[0]);
	double maxTruth = std::stod(modTokens[1]);
	double minScore = std::stod(modTokens[2]);
	double maxScore = std::stod(modTokens[3]);

	Eigen::MatrixX3i T_0(modelMesh->polygons.size(), 3);
	for (int i = 0; i < modelMesh->polygons.size(); i++) {
		pcl::Vertices _v = modelMesh->polygons[i];
		T_0(i, 0) = _v.vertices[0];
		T_0(i, 1) = _v.vertices[1];
		T_0(i, 2) = _v.vertices[2];
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr modelPc(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::fromPCLPointCloud2(modelMesh->cloud, *modelPc);
	Eigen::MatrixX3d V_0(modelMesh->cloud.width, 3);
	for (int i = 0; i < modelMesh->cloud.width; i++) {
		pcl::PointXYZ _p = modelPc->points[i];
		V_0(i, 0) = _p.x;
		V_0(i, 1) = _p.y;
		V_0(i, 2) = _p.z;
	}

	Eigen::VectorXd score(numFiles);
	QProgressDialog calBar;
	calBar.setLabelText("Scoring...");
	calBar.setRange(0, 100);
	Qt::WindowFlags calBarFlag = calBar.windowFlags();
	calBar.setWindowFlags(calBarFlag | Qt::WindowStaysOnTopHint);
	//Calculate scores of all data
	for (int j = 0; j < numFiles; j++) {
		QApplication::processEvents(); //Update UI response
		calBar.setValue(j * 100 / numFiles);
		std::stringstream idxScore;
		idxScore << std::setw(quantile) << std::setfill('0') << j + 1;
		std::string truthFileName = idxScore.str();
		pcl::PolygonMesh::Ptr testMesh(new pcl::PolygonMesh);
		if (_access((scorePath + "\\mesh_" + truthFileName + ".ply").c_str(), 0) != 0) {//mesh not exist
			score[j] = std::numeric_limits<double>::quiet_NaN();
			continue;
		}
		pcl::io::loadPLYFile(scorePath + "\\mesh_" + truthFileName + ".ply", *testMesh);
		pcl::PointCloud<pcl::PointXYZ>::Ptr testPc(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::fromPCLPointCloud2(testMesh->cloud, *testPc);
		Eigen::MatrixX3d V_S(testPc->size(), 3);
		for (int k = 0; k < testPc->size(); k++) {
			V_S(k, 0) = testPc->points[k].x;
			V_S(k, 1) = testPc->points[k].y;
			V_S(k, 2) = testPc->points[k].z;
		}
		//Translate the target point's center of gravity to the origin
		Eigen::VectorXd V_0Centroid = V_S.colwise().mean();
		V_S.rowwise() -= V_0Centroid.transpose();
		//normalization
		V_S /= V_S.norm();
		Procrustes(V_0, V_S);
		score[j] = EvaluateScore(V_0, V_S, T_0);
	}
	calBar.close();

	Eigen::VectorXd normScore = (score - Eigen::VectorXd::Constant(score.size(), minScore)) /
		(maxScore - minScore) * (maxTruth - minTruth) + Eigen::VectorXd::Constant(score.size(), minTruth);
	QTableWidget* scoreTable = ui.scoreResultTable;
	scoreTable->setColumnCount(2);
	QTableWidgetItem* scoreItem1 = new QTableWidgetItem("truthFileName");
	QTableWidgetItem* scoreItem2 = new QTableWidgetItem("score");
	scoreTable->setHorizontalHeaderItem(0, scoreItem1);
	scoreTable->setHorizontalHeaderItem(1, scoreItem2);
	while (scoreTable->rowCount() > 0)//clear the table
		scoreTable->removeRow(0);
	for (int i = 0; i < numFiles; i++) {
		QTableWidgetItem* fileNameItme = new QTableWidgetItem(truthFileName[i].c_str());
		scoreTable->insertRow(i);
		if (fileNameItme)
			scoreTable->setItem(i, 0, fileNameItme);
		QTableWidgetItem* scoreItem = new QTableWidgetItem(std::to_string(normScore[i]).c_str());
		if (scoreItem)
			scoreTable->setItem(i, 1, scoreItem);
	}
}


void LShapeAnalyser_single::exportScore() {
	QString filePath = QFileDialog::getSaveFileName(this, "Export data", "", "Text Files (*.txt);;CSV Files (*.csv);;Excel Files (*.xls *.xlsx)");
	if (!filePath.isEmpty()) {
		QFile file(filePath);

		if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
			QTextStream out(&file);

			QTableWidget* resultTable = ui.scoreResultTable;
			for (int row = 0; row < resultTable->rowCount(); ++row) {
				for (int col = 0; col < resultTable->columnCount(); ++col) {
					out << resultTable->item(row, col)->text() << ",";
				}
				out << "\n";
			}
		}
		file.close();
	}
}


LShapeAnalyser_single::~LShapeAnalyser_single()
{}