/*
 * LShapeAnalyser_multiple.cpp
 *
 *  Created on: August 30, 2023
 *      Author: Jialong Zhang
 */

#include <io.h>
#include <direct.h>
#include <chrono>
#include <random>
#include <filesystem>

#include <QFileDialog>
#include <QTextEdit>
#include <QLineEdit>
#include <QFile>
#include <QTextStream>
#include <QMessageBox>
#include <QProgressBar>
#include <QTableWidget>
#include <QDir>
#include <QComboBox>
#include <QStringList>

#include <opencv2/highgui/highgui.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/poisson.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS

#include "LShapeAnalyser_multiple.h"
#include "LShapeAnalyserFunction.h"
#include "FNNModel.h"
#include "OrbifoldTutteEmbedding.h"

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
typedef Surface_mesh::Vertex_index Vertex_index;
typedef Surface_mesh::Halfedge_index Halfedge_index;


void LShapeAnalyser_multiple::setOption(const QStringList selectOpiton) {
	option = selectOpiton;
}


LShapeAnalyser_multiple::LShapeAnalyser_multiple(QWidget* parent)
	: QWidget(parent)
{
	ui.setupUi(this);

	QString currentPath = QDir::currentPath();
	QDir dir(currentPath);
	QStringList modelFileList = dir.entryList(QStringList("*.pt"), QDir::Files);
	if (!modelFileList.isEmpty()) {
		ui.selectModelComboBox->addItems(modelFileList);
		ui.selectScoreModelComboBox->addItems(modelFileList);
	}

	connect(ui.importXButton, SIGNAL(clicked()), this, SLOT(importX()));
	connect(ui.importYButton, SIGNAL(clicked()), this, SLOT(importY()));
	connect(ui.trainButton, SIGNAL(clicked()), this, SLOT(train()));
	connect(ui.evaluateButton, SIGNAL(clicked()), this, SLOT(evaluate()));
	connect(ui.cancelButton, SIGNAL(clicked()), this, SLOT(cancel()));
	connect(ui.importDepthButton, SIGNAL(clicked()), this, SLOT(importDepth()));
	connect(ui.importDLCPathButton, SIGNAL(clicked()), this, SLOT(importDLC()));
	connect(ui.removeDepthButton, SIGNAL(clicked()), this, SLOT(removeDepth()));
	connect(ui.removeDLCPathButton, SIGNAL(clicked()), this, SLOT(removeDLC()));
	connect(ui.startScoreButton, SIGNAL(clicked()), this, SLOT(startScore()));
}


void LShapeAnalyser_multiple::cancel() {
	_stopExecutionMultiple = true;
}


void LShapeAnalyser_multiple::importX() {
	QString inputPath = QFileDialog::getOpenFileName(nullptr, "select the input data", "", "text file(*.txt)");
	QLineEdit* ql = ui.xPathLineEdit;
	if (!inputPath.isEmpty())
		ql->setText(inputPath);

	_inputData.clear();
	std::ifstream inputFile(inputPath.toStdString(), std::ios::binary);
	if (inputFile.is_open()) {
		std::string values;
		while (std::getline(inputFile, values)) {
			std::vector<double> row;
			std::istringstream iss(values);
			std::string value;
			while (std::getline(iss, value, ','))
				row.emplace_back(std::stod(value));
			_inputData.emplace_back(row);
		}
	}
	else {
		QMessageBox::warning(nullptr, "Input data file open fail", "Can't not open the input data file");
		return;
	}
	inputFile.close();
}


void LShapeAnalyser_multiple::importY() {
	QString targetPath = QFileDialog::getOpenFileName(nullptr, "select the target data", "", "text file(*.txt)");
	QLineEdit* ql = ui.yPathLineEdit;
	if (!targetPath.isEmpty())
		ql->setText(targetPath);

	_targetData.clear();
	std::ifstream targetFile(targetPath.toStdString(), std::ios::binary);
	if (targetFile.is_open()) {
		std::string value;
		while (std::getline(targetFile, value))
			_targetData.emplace_back(std::stod(value));
	}
	else {
		QMessageBox::warning(nullptr, "Target data file open fail", "Can't not open the target data file");
		return;
	}
	targetFile.close();
}


void LShapeAnalyser_multiple::train() {
	QTextEdit* showPart = ui.showPartTextEdit;
	for (int i = 0; i < option.size(); i++)
		showPart->append(option[i]);

	if (ui.xPathLineEdit->text().isEmpty() || ui.yPathLineEdit->text().isEmpty()) {
		QMessageBox::warning(nullptr, "Data empty", "Data have not import");
		return;
	}

	char mainPath[_MAX_PATH];
	if (!_getcwd(mainPath, _MAX_PATH)) {
		QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
		return;
	}


	if (_inputData.size() != _targetData.size()) {
		QMessageBox::warning(nullptr, "Inconsistent size", "Inconsistent size between input and target data");
		return;
	}

	int inputSize = _inputData[0].size();
	int dataVolume = _inputData.size();
	torch::Tensor inputTensor = torch::empty({ dataVolume,inputSize });
	for (int i = 0; i < dataVolume; ++i)
		for (int j = 0; j < inputSize; ++j)
			inputTensor[i][j] = _inputData[i][j];

	torch::Tensor targetTensor = torch::tensor(_targetData);

	if (inputSize != option.size()) {
		QMessageBox::warning(nullptr, "Input size error", "The size of input data is not consistent with the selected option number");
		return;
	}

	double learningRate = ui.lrLineEdit->text().toDouble();
	int numEpochs = ui.epochSpinBox->value();

	// Define models and optimizers
	FNNModelImpl model(option.size(), 10, 1);
	model.train();

	// Define loss functions and optimizers
	torch::nn::MSELoss criterion;
	torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(learningRate).momentum(0.5));

	// model training
	QProgressBar* bar = ui.progressBar;
	for (int epoch = 0; epoch < numEpochs; epoch++)
	{
		// forward propagation
		QApplication::processEvents();
		if (_stopExecutionMultiple) {
			QMessageBox::information(nullptr, "Cancel", "User cancel the evaluation");
			bar->setValue(0);
			_stopExecutionMultiple = false;
			return;
		}
		bar->setValue(epoch * 100 / numEpochs);
		torch::Tensor output = model.forward(inputTensor);
		torch::Tensor loss = criterion(output, targetTensor);

		// Backpropagation and optimization
		optimizer.zero_grad();
		loss.backward();
		optimizer.step();
	}
	QString modelName = ui.modelNameLineEdit->text();
	
	torch::serialize::OutputArchive output_archive;
	model.save(output_archive);
	output_archive.save_to(std::string(mainPath) + "\\" + modelName.toStdString() + ".pt");

	QString currentPath = QDir::currentPath();
	QDir dir(currentPath);
	QStringList modelFileList = dir.entryList(QStringList("*.pt"), QDir::Files);
	ui.selectModelComboBox->clear();
	if (!modelFileList.isEmpty()) {
		ui.selectModelComboBox->addItems(modelFileList);
		ui.selectScoreModelComboBox->addItems(modelFileList);
	}
	bar->setValue(100);
	QMessageBox::information(nullptr, "Train complete", "Train complete");
}


void LShapeAnalyser_multiple::evaluate() {
	char mainPath[_MAX_PATH];
	if (!_getcwd(mainPath, _MAX_PATH)) {
		QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
		return;
	}

	if (ui.selectModelComboBox->currentText().isEmpty()) {
		QMessageBox::warning(nullptr, "Not model exist or select", "Not model exist or select");
		return;
	}

	std::vector<std::string> meshPath;
	std::vector<_finddatai64_t> fileInfo;
	std::vector< intptr_t> fileHandle;
	for (int i = 0; i < option.size(); i++) {
		meshPath.emplace_back(std::string(mainPath) + "\\" + option[i].toStdString() + "\\mesh");
		_finddatai64_t info;
		intptr_t handle;
		if (_access(meshPath[i].c_str(), 0) != 0 || (handle = _findfirst64((meshPath[i] + "\\*.ply").c_str(), &info)) == -1) {
			QMessageBox::warning(nullptr, "Mesh data not exist", "Mesh data does not exist, please reconstruct the mesh firstly");
			return;
		}
		fileInfo.emplace_back(info);
		fileHandle.emplace_back(handle);
	}

	// Calculate the number of files
	Eigen::VectorXi fileNumber = Eigen::VectorXi::Zero(option.size());
	for (int i = 0; i < option.size(); i++) {
		for (const auto& entry : std::filesystem::directory_iterator(meshPath[i]))
			if (std::filesystem::is_regular_file(entry) && entry.path().extension() == ".ply")
				fileNumber[i]++;

		if (i > 0 && fileNumber[i] != fileNumber[static_cast<int64_t>(i) - 1]) {
			QMessageBox::warning(nullptr, "File number is inconsistent", "File number of each option is inconsistent!");
			return;
		}
	}

	std::vector<int> fileIdx;
	for (int i = 0; i < fileNumber[0]; i++)
		fileIdx.emplace_back(i);
	
	int trainNum = fileNumber[0] * 0.8;//The size of  data set used to train
	int testNum = fileNumber[0] - trainNum; //The size of data set used to test

	// Read truth value
	std::string truthFilePath = std::string(mainPath) + "\\" + option[0].toStdString() + "\\Truth.csv";
	if (_access(truthFilePath.c_str(), 0) != 0) {
		QMessageBox::warning(nullptr, "Not file exist", "Truth value file is not exist");
		return;
	}

	std::vector<std::vector<std::string>> remainFile;
	for (int i = 0; i < option.size(); i++) {
		std::vector<std::string > remain;
		do {
			std::string number = fileInfo[i].name;
			remain.emplace_back(number);
		} while (_findnext64(fileHandle[i], &fileInfo[i]) == 0);
		remainFile.emplace_back(remain);
		if (i > 0) {
			for (int j = 0; j < remain.size(); j++) {
				if (remainFile[i][j] != remainFile[static_cast<int64_t>(i) - 1][j]) {
					QMessageBox::warning(nullptr, "Inconsistent number", "Inconsistent number from data" + option[i] + " to " + option[i-1]);
					return;
				}
			}
		}
	}

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

			int pos2 = 0;
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
		auto it = std::find(remainFile[0].begin(), remainFile[0].end(), truthFileName[i]);
		if (it == remainFile[0].end()) {
			truthFileName.erase(truthFileName.begin() + i);
			truthValue.erase(truthValue.begin() + i);
		}
	}

	// Load the model
	FNNModelImpl model(option.size(), 10, 1);
	std::string modelPath = std::string(mainPath) + "\\" + ui.selectModelComboBox->currentText().toStdString();
	torch::serialize::InputArchive input_archive;
	input_archive.load_from(modelPath);
	model.load(input_archive);
	model.eval();

	int timesFold = ui.testTimesSpinBox->value();
	Eigen::VectorXd accuracy(timesFold);
	Eigen::MatrixX4d approximateAccuracy(timesFold, 4);
	QProgressBar* bar = ui.progressBar;
	for (int i = 0; i < timesFold; i++) {
		//Determine if the user has interrupted the program
		QApplication::processEvents(); //Update UI response
		if (_stopExecutionMultiple) {
			QMessageBox::information(nullptr, "Cancel", "User cancel the evaluation");
			bar->setValue(0);
			_stopExecutionMultiple = false;
			return;
		}
		bar->setValue(i * 100 / timesFold);

		//Obtain the training set randomly from whole set
		std::vector<int> trainIdx = fileIdx;
		std::vector<int> testIdx;

		std::random_device rd;
		std::mt19937 rng(rd());
		// Shuffle the randomVector
		std::shuffle(trainIdx.begin(), trainIdx.end(), rng);

		// Resize the randomVector to contain only n elements
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

		//QMessageBox::information(nullptr, "", "test");
		Eigen::VectorXd enumValue;
		torch::Tensor normScoreTensor = torch::empty({ testNum,option.size() });
		for (int op = 0; op < option.size(); op++) {
			pcl::PolygonMesh::Ptr trainMesh(new pcl::PolygonMesh);//average shape variable
			pcl::io::loadPLYFile((std::string(mainPath) + "\\" + option[op].toStdString() + "\\mesh" + "\\" + trainFileName[0]).c_str(), *trainMesh);
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
				if (_access((meshPath[0] + "\\" + trainFileName[j]).c_str(), 0) != 0)
					QMessageBox::warning(nullptr, "File not exist", "Check if all options have the same number of meshes and corresponding file names.");
				pcl::io::loadPLYFile(meshPath[0] + "\\" + trainFileName[j], *trainMesh);
				pcl::PointCloud<pcl::PointXYZ>::Ptr trainPc(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::fromPCLPointCloud2(trainMesh->cloud, *trainPc);
				Eigen::MatrixX3d trainVertices(trainPc->size(), 3);
				for (int k = 0; k < trainPc->size(); k++) {
					trainVertices(k, 0) = trainPc->points[k].x;
					trainVertices(k, 1) = trainPc->points[k].y;
					trainVertices(k, 2) = trainPc->points[k].z;
				}
				Eigen::VectorXd centroid_v = trainVertices.colwise().mean();
				trainVertices.rowwise() -= centroid_v.transpose();
				trainVertices /= trainVertices.norm();
				V.emplace_back(trainVertices);
			}

			//Alignment
			int iter = 0;
			while (iter < 10) {
				iter++;

				//Align all shapes to the average shape
				for (int j = 0; j < trainNum; j++) {
					Procrustes(V_0, V[j]);
				}
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
			V_0 = (V_0.rowwise() - centroid.transpose()) * eigenVectors * rotateX;
			for (int j = 0; j < trainNum; j++)
				V[j] = (V[j].rowwise() - centroid.transpose()) * eigenVectors * rotateX;

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
				pcl::io::loadPLYFile(meshPath[0] + "\\" + testFileName[j], *testMesh);
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
				testScore[j] = EvaluateScore(V_0, V_S, T_0);
			}

			//Standardized truth
			double up = ui.upperLimitLineEdit->text().toDouble();;
			double low = ui.lowerLimitLineEdit->text().toDouble();
			if (up < low) {
				QMessageBox::warning(nullptr, "limit error", "The lower limit must be less than the upper");
				return;
			}
			double current = low;
			double step = ui.discreIncreLineEdit->text().toDouble();

			if (enumValue.size() == 0) {
				enumValue.resize(int((up - low) / step + 1));
				int idx = 0;
				while (current <= up) {
					enumValue[idx] = current;
					current += step;
					idx++;
				}
			}
			Eigen::VectorXd normScore = ((testScore.array() - minScore).array() / (maxScore - minScore) * (maxTruth - minTruth)).array() + minTruth;
			for (int ns = 0; ns < testNum; ns++)
				for (int optionIndex = 0; optionIndex < option.size(); optionIndex++)
					normScoreTensor[ns][optionIndex] = normScore[ns];
		}

		// Calculate accuracy
		Eigen::VectorXd predictScore(testNum);
		torch::Tensor outputScore = model.forward(normScoreTensor);
		for (int predictIdx = 0; predictIdx < testNum; predictIdx++)
			predictScore[predictIdx] = outputScore[predictIdx].item<double>();

		int accuracyNum = 0;
		Eigen::Vector4d approximateNum = Eigen::Vector4d::Zero();

		for (int j = 0; j < testNum; j++) {
			// Discretizing Truth
			if (ui.discreIncreLineEdit->text().toDouble() != 0) {
				Eigen::VectorXd neigh = (enumValue.array() - predictScore[j]).abs();
				auto it = std::find(neigh.begin(), neigh.end(), neigh.minCoeff());
				int enumIdx = it - neigh.begin();
				predictScore[j] = enumValue[enumIdx];
			}
			if (std::abs(predictScore[j] - truthValueTest[j]) <= 0)
				accuracyNum++;

			double disIncre = ui.discreIncreLineEdit->text().toDouble();
			for (int k = 0; k < approximateNum.size(); k++)
				if (std::abs(predictScore[j] - truthValueTest[j]) <= disIncre * (k+1))
					approximateNum[k] += 1;
		}
		accuracy[i] = double(accuracyNum) / double(testNum);
		approximateAccuracy.row(i) = approximateNum / double(testNum);
	}
	bar->setValue(100);
	Eigen::VectorXd Avg(5);
	Eigen::VectorXd Min(5);
	Eigen::VectorXd Max(5);
	for (size_t i = 0; i < 5; i++) {
		if (i == 0) {
			Avg[i] = accuracy.mean() * 100;
			Min[i] = accuracy.minCoeff() * 100;
			Max[i] = accuracy.maxCoeff() * 100;
		}
		else {
			Avg[i] = approximateAccuracy.col(i - 1).mean() * 100;
			Min[i] = approximateAccuracy.col(i - 1).minCoeff() * 100;
			Max[i] = approximateAccuracy.col(i - 1).maxCoeff() * 100;
		}
	}
	QTableWidget* evaluateTable = ui.resultTable;

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


void LShapeAnalyser_multiple::importDepth() {
	QString filePath = QFileDialog::getOpenFileName(this, "Select depth image file", "", "depth image file(*.png)");
	if (!filePath.isEmpty()) {
		_depthPath.emplace_back(filePath);
		ui.depthComboBox->addItem(filePath);
	}
}

void LShapeAnalyser_multiple::importDLC() {
	QString DLCFolder = QFileDialog::getExistingDirectory(this, "Select depth image file", "", QFileDialog::ShowDirsOnly);
	if (!DLCFolder.isEmpty()) {
		if (_access((DLCFolder.toStdString() + "\\config.yaml").c_str(), 0) == 0) {
			_DLCPath.emplace_back(DLCFolder);
			ui.DLCPathComboBox->addItem(DLCFolder);
		}
		else {
			QMessageBox::warning(nullptr, "DLC folder path error", "DLC folder path error");
			return;
		}
	}
}


void LShapeAnalyser_multiple::removeDepth() {
	QString depthPath = ui.depthComboBox->currentText();
	if (!depthPath.isEmpty()) {
		QMessageBox::StandardButton reply;
		reply = QMessageBox::question(this, "remove", "Remove depth path " + depthPath + "?", QMessageBox::Yes | QMessageBox::No);
		if (reply == QMessageBox::Yes) {
			int currentIdx = ui.depthComboBox->currentIndex();
			ui.depthComboBox->removeItem(currentIdx);
			_depthPath.erase(currentIdx + _depthPath.begin());
			QMessageBox::information(nullptr, "Remove successfully", "Remove depth successfully");
		}
	}
}


void LShapeAnalyser_multiple::removeDLC() {
	QString DLCPath = ui.DLCPathComboBox->currentText();
	if (!DLCPath.isEmpty()) {
		QMessageBox::StandardButton reply;
		reply = QMessageBox::question(this, "remove", "Remove DLC path " + DLCPath + "?", QMessageBox::Yes | QMessageBox::No);
		if (reply == QMessageBox::Yes) {
			int currentIdx = ui.depthComboBox->currentIndex();
			ui.depthComboBox->removeItem(currentIdx);
			_DLCPath.erase(currentIdx + _DLCPath.begin());
			QMessageBox::information(nullptr, "Remove successfully", "Remove depth successfully");
		}
	}
}


void LShapeAnalyser_multiple::startScore() {
	char mainPath[_MAX_PATH];
	if (!_getcwd(mainPath, _MAX_PATH)) {
		QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
		return;
	}

	if (ui.selectScoreModelComboBox->currentText().isEmpty()) {
		QMessageBox::warning(nullptr, "Not model exist or select", "Not model exist or select");
		return;
	}

	if (ui.depthComboBox->count() != option.size()) {
		QMessageBox::warning(nullptr, "Depth data not correct", "Depth data are not equal to the option number");
		return;
	}

	if (ui.DLCPathComboBox->count() != option.size()) {
		QMessageBox::warning(nullptr, "DLC path not correct", "Depth data are not equal to the option number");
		return;
	}

	std::string scorePath = std::string(mainPath) + "\\scoreMulTemp";
	if (_access(scorePath.c_str(), 0) == 0) {
		DeleteDiretoryContents(scorePath);
		std::filesystem::remove(scorePath);
	}
	if (_mkdir(scorePath.c_str()) != 0) { //create a temporary folder to store intermediate file
		QMessageBox::warning(nullptr, "Folder gray create failure", "Folder gray create failure");
		return;
	}

	// calibration parameters
	double fx = ui.fxLineEdit->text().toDouble();
	double fy = ui.fyLineEdit->text().toDouble();
	double cx = ui.cxLineEdit->text().toDouble();
	double cy = ui.cyLineEdit->text().toDouble();

	torch::Tensor normScoreTensor = torch::empty({ 1,option.size() });
	for (int op = 0; op < option.size(); op++) {
		//write video
		cv::VideoWriter writer;
		cv::Mat depthImage = cv::imread(_depthPath[op].toStdString(), cv::IMREAD_UNCHANGED);
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
				grayImage.at<uchar>(row, col) = (depthImage.at<unsigned short>(row, col) - depthMin) / (depthMax - depthMin) * 255.0;
		std::string grayTargetFile = scorePath + "\\gray.png";
		cv::imwrite(grayTargetFile, grayImage);
		writer = cv::VideoWriter(scorePath + "\\Synthetic.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 32, cv::Size(h, w), true);
		cv::Mat image = cv::imread(grayTargetFile);
		writer << image;
		writer.release();


		//DLC
		std::string DLCPath = ui.DLCPathComboBox->itemText(op).toStdString();
		std::string videoFilePath = scorePath + "\\Synthetic.avi";
		std::ofstream pyFile(scorePath + "\\DLC_analyze.py");
		if (!pyFile.is_open()) {
			QMessageBox::warning(nullptr, "file create fail", "file create fail");
			return;
		}
		std::string code1 = "import deeplabcut";
		std::string code2 = "config_path=r\"" + DLCPath + "\\config.yaml\"";
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


		// convert to mesh
		pcl::PolygonMesh::Ptr modelMesh(new pcl::PolygonMesh);
		pcl::io::loadPLYFile((std::string(mainPath) + "\\" + option[op].toStdString() + "\\model.ply").c_str(), *modelMesh);

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

		std::ifstream modelFile(std::string(mainPath) + "/" + option[op].toStdString() + "/modelKeyPointIndex.txt");
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
		cv::Mat depthImg = cv::imread(ui.depthComboBox->itemText(op).toStdString(), cv::IMREAD_UNCHANGED);
		int row = depthImg.rows;
		int col = depthImg.cols;

		std::vector<double> cornerCoordinate = markPoint[0];
		int keyPointNum = markPoint[0].size() / 2;

		std::vector<Point_2d> keyPoint;

		bool isDepthMissing = false;
		for (int i = 0; i < keyPointNum; i++) {
			Point_2d nullPoint = { 0,0 };
			keyPoint.emplace_back(nullPoint);
			keyPoint[i].x = std::floor(cornerCoordinate[2 * static_cast<double>(i)]);
			keyPoint[i].y = std::floor(cornerCoordinate[2 * static_cast<double>(i) + 1]);
			if (depthImg.at<uint16_t>(keyPoint[i].y, keyPoint[i].x) == 0) {
				isDepthMissing = true;
				break;
			}
		}
		if (isDepthMissing) {
			QMessageBox::warning(nullptr, "Depth value is miss", "Depth value miss, can't score");
			return;
		}
		bool isKeyDevia = false;
		for (int i = 0; i < keyPointNum; i++) {
			if (likelihood[0][i] < 0.95) {
				isKeyDevia = true;
				break;
			}
		}
		if (isKeyDevia) {
			QMessageBox::warning(nullptr, "Key point deviation", "Key points are deviated, can't score");
			return;
		}

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
		verticesKeyPointI->clear();
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

		cgalMesh.clear();
		for (const auto& point : pcSource->points)
			cgalMesh.add_vertex(Point_3(point.x, point.y, point.z));
		for (const auto& polygon : mesh->polygons)
			cgalMesh.add_face(Vertex_index(polygon.vertices[0]), Vertex_index(polygon.vertices[1]),
				Vertex_index(polygon.vertices[2]));
		halfEdges = cgalMesh.halfedges();

		// get the free boundary
		for (auto it = halfEdges.begin(); it != halfEdges.end(); it++)
			if (cgalMesh.is_border(*it)) {
				boundarySource.conservativeResize(boundarySource.size() + 1);
				Vertex_index boundaryIndex = cgalMesh.source(*it);
				boundarySource[boundarySource.size() - 1] = boundaryIndex;
			}

		nextBoundaryVertex.clear();
		for (auto it = halfEdges.begin(); it != halfEdges.end(); it++)
			if (cgalMesh.is_border(*it)) {
				Vertex_index boundaryIndex = cgalMesh.source(*it);
				Vertex_index nextBoundaryIndex = cgalMesh.target(*it);
				nextBoundaryVertex[boundaryIndex] = nextBoundaryIndex;
			}

		Vertex_index boundarySourceStart = Vertex_index(boundarySource[0]);
		index = 0;
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
		nearestKeyPoint1Index = Dsearchn(pcSource, boundarySource, verticesKeyPointI->at(0));
		nearestKeyPoint1Head = boundarySource.head(nearestKeyPoint1Index);
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
			QMessageBox::warning(nullptr, "Can't not map", "Can't not map");
			return;
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

		//score
		//Read model parametersand load model shape
		std::string modelCsvPath = std::string(mainPath) + "\\" + option[op].toStdString() + "\\model.csv";
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

		double score;
		//Calculate scores of all data
		std::stringstream idxScore;
		std::string truthFileName = idxScore.str();
		pcl::PolygonMesh::Ptr testMesh(new pcl::PolygonMesh);
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
		score = EvaluateScore(V_0, V_S, T_0);
		double normScore = (score - minScore) / (maxScore - minScore) * (maxTruth - minTruth) + minTruth;
		normScoreTensor[0][op] = normScore;
	}

	FNNModelImpl model(option.size(), 10, 1);
	std::string modelPath = std::string(mainPath) + "\\" + ui.selectScoreModelComboBox->currentText().toStdString();
	torch::serialize::InputArchive input_archive;
	input_archive.load_from(modelPath);
	model.load(input_archive);
	model.eval();

	torch::Tensor outputScore = model.forward(normScoreTensor);
	double predictScore = outputScore[0].item<double>();
	QLineEdit* outputValueLineEdit = ui.outputValueLineEdit;
	outputValueLineEdit->setText(std::to_string(predictScore).c_str());
}


LShapeAnalyser_multiple::~LShapeAnalyser_multiple()
{}