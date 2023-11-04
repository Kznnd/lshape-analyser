/*
 * LShapeAnalyser.h
 *
 *  Created on: August 30, 2023
 *      Author: Jialong Zhang
 */

#pragma once

#include <QtWidgets/QWidget>

#include "ui_LShapeAnalyser.h"
#include "LShapeAnalyser_single.h"
#include "LShapeAnalyser_multiple.h"

class LShapeAnalyser : public QWidget
{
    Q_OBJECT

public:
    LShapeAnalyser(QWidget *parent = nullptr);
    ~LShapeAnalyser();

private:
    Ui::LShapeAnalyserClass ui;

private slots:
    void createNewButton();
    void loadOldButton();
    void renameOldButton();
    void deleteOldButton();
    void startMultipleButton();
};
