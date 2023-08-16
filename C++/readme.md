###  **The tool of C++ version is developing.** 

# **LShapeAnalyser_C++** #
Created by **Jialong Zhang(张嘉龙)** from China Agricultural University.  
# Introdution #
LShapeAnalyser tool written in C++.
# Environment Requirements #
**Visual Studio 2019, PCL-1.12.1, Qt-5.14.2, OpenCV-4.6.0, CGAL-5.6**
# Environment Preparation #
## Step1 ##
Install [anaconda](https://www.anaconda.com/data-science-platform).
## Step2 ##
Install [DeepLabCut 2.3.3](https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/installation.md) with the methods of **CONDA**.
## Step3 ##
Download the [pcl-1.12.1](https://github.com/PointCloudLibrary/pcl/releases).
## Step4 ##
Download [Qt-5.14.2](https://download.qt.io/archive/qt/5.14/5.14.2/).
## Step5 ##
Download [OpenCV-4.6.0](https://sourceforge.net/projects/opencvlibrary/files/4.6.0/opencv-4.6.0-vc14_vc15.exe/download).
## Step6 ##
Download [CGAL-5.6](https://github.com/CGAL/cgal/releases).
# Usage #
Users can directly compile and run this project in Visual studio 2019.
## Select mode ##
User can select a mode through the tab(single or multiple) in the main page.
# Singel mode #
##(A)create or load a project##
User can input the project name in the edit field in the panel "Create new project" to create a new projcet or select a old project then load in the panel "Operate old project" where can also do some other management operations.  
![Main_page](Fig/Main_page.png)  
**Main page**  
##(B)Import depth image##
After create or load a project, a new page--the project page will open. User can select 'File' in the menu bar, and then select 'Import'. A folder selection interface will pop up. Please select a folder containing the depth images you want to use (this will import all files with the extension png under this folder, so do not place other images in this folder).Then it will convert the imported depth images into grayscale images and synthesize a video with the extension avi one by one frame(used as input for DeepLabCut(DLC)).
