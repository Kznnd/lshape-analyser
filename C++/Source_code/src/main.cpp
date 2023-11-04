/*
 * main.cpp
 *
 *  Created on: August 30, 2023
 *      Author: Jialong Zhang
 */

#include "LShapeAnalyser.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    LShapeAnalyser lshape;
    lshape.setWindowTitle("Main");
    lshape.show();
    return a.exec();
}
