/*
 * LShapeAnalyser.cpp
 *
 *  Created on: August 30, 2023
 *      Author: Jialong Zhang
 */

#include <direct.h>
#include <io.h>
#include <fstream>
#include <filesystem>

#include <QMessageBox>
#include <QTextEdit>
#include <QComboBox>
#include <QListWidget>
#include <QDir>

#include "LShapeAnalyser.h"
#include "LShapeAnalyserFunction.h"

LShapeAnalyser::LShapeAnalyser(QWidget* parent)
	: QWidget(parent)
{
	ui.setupUi(this);

	char mainPath[_MAX_PATH];
	if (!_getcwd(mainPath, _MAX_PATH)) {//Get the main path
		QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
		return;
	}

	//Read the project list from project.txt file and import it into the control
	std::string filePath = std::string(mainPath) + "\\project.txt";
	if (_access(filePath.c_str(), 0) == 0) {// Confirm file existence
		std::ifstream file(filePath);
		std::string lineStr;
		QComboBox* selectProject = ui.selectProjectComboBox;
		QListWidget* optionList = ui.optionList;
		while (std::getline(file, lineStr)) {
			selectProject->addItem(lineStr.c_str());
			optionList->addItem(lineStr.c_str());
		}
	}

	//Signal Connection
	connect(ui.createNewButton, SIGNAL(clicked()), this, SLOT(createNewButton()));
	connect(ui.loadOldButton, SIGNAL(clicked()), this, SLOT(loadOldButton()));
	connect(ui.renameOldButton, SIGNAL(clicked()), this, SLOT(renameOldButton()));
	connect(ui.deleteOldButton, SIGNAL(clicked()), this, SLOT(deleteOldButton()));
	connect(ui.startMultipleButton, SIGNAL(clicked()), this, SLOT(startMultipleButton()));
}


void LShapeAnalyser::createNewButton() {
	QTextEdit* projectNameText = ui.projectNameText;
	QString projectName = projectNameText->toPlainText();
	if (projectName.isEmpty()) {
		QMessageBox::warning(nullptr, "Project name empty error", "Project name is empty");
		return;
	}

	QMessageBox::StandardButton reply;
	reply = QMessageBox::question(this, "create", "Create project with name " + projectName, QMessageBox::Yes | QMessageBox::No);
	if (reply == QMessageBox::Yes) {
		char mainPath[_MAX_PATH];
		if (!_getcwd(mainPath, _MAX_PATH)) {// Get the main path
			QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
			return;
		}
		std::string filePath = std::string(mainPath) + "\\project.txt";
		if (!_access(filePath.c_str(), 0) == 0) { // The project.txt file is not exist
			std::ofstream file(filePath); // Create a file named project.txt
			if (file.is_open()) {
				file << projectName.toStdString(); // write the project name in file
				file.close();
			}
			else {
				QMessageBox::warning(nullptr, "Open file error", "Can't open file 'project.txt'");
				return;
			}
		}
		else {//The project.txt file is exist
			std::ifstream file(filePath);
			std::string lineStr;
			while (std::getline(file, lineStr)) {//Search the project name whether exist
				if (lineStr == projectName.toStdString()) {
					QMessageBox::warning(nullptr, "Project already exist error", "Project already exist");
					return;
				}
			}

			//Add project name in file
			std::ofstream fileApp(filePath, std::ios::app);
			if (fileApp.is_open()) {
				fileApp << "\n" + projectName.toStdString();
				fileApp.close();
			}
			else {
				QMessageBox::warning(nullptr, "Open file error", "Can't open file 'project.txt'");
				return;
			}
		}

		//create the corresponding project folder
		if (_mkdir((std::string(mainPath) + "\\" + projectName.toStdString()).c_str()) != 0) {
			QMessageBox::warning(nullptr, "Creaet folder error", "Can't create folder" + projectName);
			return;
		}

		//update the Main window
		QComboBox* selectProjectComboBox = ui.selectProjectComboBox;
		QListWidget* optionList = ui.optionList;
		selectProjectComboBox->addItem(projectName);
		optionList->addItem(projectName);

		//create a children window
		LShapeAnalyser_single* LShapeSingle = new LShapeAnalyser_single(this);
		LShapeSingle->setWindowTitle("Single_" + projectName);
		LShapeSingle->setWindowModality(Qt::ApplicationModal); //Set as Modal Window
		LShapeSingle->setProjectName(projectName);
		LShapeSingle->setDLCPath();
		LShapeSingle->setAttribute(Qt::WA_DeleteOnClose);//Delete the object after close the window to avoid memory leak
		LShapeSingle->show();
	}
	else { //reply is no
		return;
	}
}


void LShapeAnalyser::loadOldButton() {
	QComboBox* selectProjectComboBox = ui.selectProjectComboBox;
	QString projectName = selectProjectComboBox->currentText();
	if (projectName.isEmpty()) {
		QMessageBox::warning(nullptr, "Project name empty error", "Project name is empty");
		return;
	}

	QMessageBox::StandardButton reply;
	reply = QMessageBox::question(this, "load", "Load project with name " + projectName, QMessageBox::Yes | QMessageBox::No);
	if (reply == QMessageBox::Yes) {
		//create a children window
		LShapeAnalyser_single* LShapeSingle = new LShapeAnalyser_single();
		LShapeSingle->setWindowTitle("Single_" + projectName);
		LShapeSingle->setWindowModality(Qt::ApplicationModal); //Set as Modal Window
		LShapeSingle->setProjectName(projectName);
		LShapeSingle->setDLCPath();
		LShapeSingle->setAttribute(Qt::WA_DeleteOnClose);//Delete the object after close the window to avoid memory leak
		LShapeSingle->show();
	}
	else {//reply==No
		return;
	}
}


void LShapeAnalyser::renameOldButton() {
	QString projectName = ui.selectProjectComboBox->currentText();
	if (projectName.isEmpty()) {
		QMessageBox::warning(nullptr, "Not project exist", "Not project exist");
		return;
	}

	QString newName = ui.newNameTextEdit->toPlainText();
	if (newName.isEmpty()) {
		QMessageBox::warning(nullptr, "Not input error", "Not input new name in textedit");
		return;
	}

	QMessageBox::StandardButton reply;
	reply = QMessageBox::question(this, "rename", "Rename project with name " + projectName, QMessageBox::Yes | QMessageBox::No);
	if (reply == QMessageBox::Yes) {
		char mainPath[_MAX_PATH];
		if (!_getcwd(mainPath, _MAX_PATH)) {//Get the main path
			QMessageBox::warning(nullptr, "Get main path failure", "Get main path failure");
			return;
		}

		//update the file project.txt
		std::string filePath = std::string(mainPath) + "\\project.txt";
		std::fstream file(filePath);
		std::string lineStr;
		std::vector<std::string> newStr;
		while (std::getline(file, lineStr)) {
			if (lineStr == projectName.toStdString()) {
				newStr.emplace_back(newName.toStdString());
			}
			else {
				newStr.emplace_back(lineStr);
			}
		}
		file.close();
		std::ofstream ofile(filePath, std::ofstream::out);
		if (ofile.is_open()) {
			for (int i = 0; i < newStr.size(); i++) {
				if (i == 0)
					ofile << newStr[i];
				else
					ofile << "\n" + newStr[i];
			}
			ofile.close();
		}
		else {
			QMessageBox::warning(nullptr, "Open file error", "Can't open file 'project.txt'");
			return;
		}

		//rename the corresponding project folder
		std::filesystem::path oldPath(std::string(mainPath) + "\\" + projectName.toStdString());
		std::filesystem::path newPath(std::string(mainPath) + "\\" + newName.toStdString());
		std::filesystem::rename(oldPath, newPath);

		//reset the select box in single tab
		QComboBox* selectProjectComboBox = ui.selectProjectComboBox;
		selectProjectComboBox->setItemText(selectProjectComboBox->currentIndex(), newName);

		//reset the option list in mutiple tab
		QListWidget* optionList = ui.optionList;
		QList<QListWidgetItem*> items = optionList->findItems(projectName, Qt::MatchExactly);
		QListWidgetItem* item = items.first();
		item->setText(newName);

		QMessageBox::information(nullptr, "Rename info", "Rename successfully");
	}
	else {
		return;
	}
}


void LShapeAnalyser::deleteOldButton() {
	QComboBox* selectProjectComboBox = ui.selectProjectComboBox;
	QString projectName = selectProjectComboBox->currentText();
	if (projectName.isEmpty()) {
		QMessageBox::warning(nullptr, "Project empty", "Project is empty");
		return;
	}

	QMessageBox::StandardButton reply;
	reply = QMessageBox::question(this, "Delete", "Delete project with name" + projectName, QMessageBox::Yes | QMessageBox::No);
	if (reply == QMessageBox::Yes) {
		char mainPath[_MAX_PATH];
		if (!_getcwd(mainPath, _MAX_PATH)) {//get the main path
			QMessageBox::warning(nullptr, "Not file", "Not file");
			return;
		}

		//update the file project.txt
		std::string filePath = std::string(mainPath) + "\\project.txt";
		std::ifstream ifile(filePath);
		std::string lineStr;
		std::vector<std::string> fileContain;
		while (std::getline(ifile, lineStr)) {
			if (projectName.toStdString() != lineStr) {
				fileContain.emplace_back(lineStr);
			}
		}
		ifile.close();

		std::ofstream ofile(filePath);
		for (int i = 0; i < fileContain.size(); i++) {
			if (i == 0)
				ofile << fileContain[i];
			else
				ofile << "\n" + fileContain[i];
		}
		ofile.close();

		// remove file project.txt if empty
		if (fileContain.size() == 0) {
			std::filesystem::remove(filePath);
		}

		//update the main window control
		selectProjectComboBox->removeItem(selectProjectComboBox->currentIndex());
		QListWidget* optionList = ui.optionList;
		QList<QListWidgetItem*> items = optionList->findItems(projectName, Qt::MatchExactly);
		QListWidgetItem* item = items.first();
		delete item;
		std::string projectPath = std::string(mainPath) + "\\" + projectName.toStdString();
		if (_access(projectPath.c_str(), 0) == 0) {
			DeleteDiretoryContents(projectPath);
			std::filesystem::remove(projectPath);
		}
		QMessageBox::information(this, "Delete", "Delete project successfully");
	}
	else {
		return;
	}
}

void LShapeAnalyser::startMultipleButton() {
	QListWidget* optionList = ui.optionList;
	QList<QListWidgetItem*> selectedItems = optionList->selectedItems();
	if (selectedItems.count() < 2) {
		QMessageBox::warning(nullptr, "Option not enough", "Selected options are not enough(At least two)");
		return;
	}

	QMessageBox::StandardButton reply;
	reply = QMessageBox::question(this, "Start mutiple", "Start the mutiple project?", QMessageBox::Yes | QMessageBox::No);
	if (reply == QMessageBox::Yes) {
		//create a children window of multiple
		LShapeAnalyser_multiple* LShapeMultiple = new LShapeAnalyser_multiple();
		std::string projectName;
		QStringList option;
		foreach(QListWidgetItem * Item, selectedItems) {
			projectName = projectName + Item->text().toStdString() + "_";
			option.append(Item->text());
		}
		LShapeMultiple->setWindowTitle(("Multiple_" + projectName).c_str());
		LShapeMultiple->setWindowModality(Qt::ApplicationModal); //Set as Modal Window
		LShapeMultiple->setOption(option);
		LShapeMultiple->setAttribute(Qt::WA_DeleteOnClose);//Delete the object after close the window to avoid memory leak
		LShapeMultiple->show();
	}
	else {//reply==No
		return;
	}
}


LShapeAnalyser::~LShapeAnalyser()
{}