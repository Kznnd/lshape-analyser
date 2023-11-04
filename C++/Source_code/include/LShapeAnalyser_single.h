/*
 * LShapeAnalyser_single.h
 *
 *  Created on: August 30, 2023
 *      Author: Jialong Zhang
 */

#pragma once

#include <QMainWindow>
#include "ui_LShapeAnalyser_single.h"

class LShapeAnalyser_single : public QMainWindow
{
	Q_OBJECT

public:
	LShapeAnalyser_single(QWidget *parent = nullptr);
	~LShapeAnalyser_single();

	void setProjectName(const QString);
	void setDLCPath();
	QString _projectName;

private:
	Ui::LShapeAnalyser_singleClass ui;
	bool _stopExecution = false;

private slots:
	void exitSingle();
	void importDepth();
	void openDLC();
	void startConvert();
	void cancel();
	void loadTruth();
	void resetTruth();
	void importTruth();
	void saveTruth();
	void startEvaluate();
	void train();
	void browseDLC();
	void editPara();
	void score();
	void exportScore();
};
