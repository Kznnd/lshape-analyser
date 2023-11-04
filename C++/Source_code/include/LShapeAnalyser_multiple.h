/*
 * LShapeAnalyser_multiple.h
 *
 *  Created on: August 30, 2023
 *      Author: Jialong Zhang
 */

#pragma once

#include <QWidget>
#include "ui_LShapeAnalyser_multiple.h"

class LShapeAnalyser_multiple : public QWidget
{
	Q_OBJECT

public:
	LShapeAnalyser_multiple(QWidget *parent = nullptr);
	~LShapeAnalyser_multiple();

private:
	QStringList option;

public:
	void setOption(const QStringList);

private:
	Ui::LShapeAnalyser_multipleClass ui;
	std::vector<std::vector<double>> _inputData;
	std::vector<double> _targetData;
	std::vector<QString> _depthPath;
	std::vector<QString> _DLCPath;
	bool _stopExecutionMultiple = false;

private slots:
	void cancel();
	void importX();
	void importY();
	void train();
	void evaluate();
	void importDepth();
	void importDLC();
	void removeDepth();
	void removeDLC();
	void startScore();
};