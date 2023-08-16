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
![Import_depth](Fig/Import_depth.png)
 **Import the depth image**   
![gray_and_video](Fig/gray_and_video.png)
**Converted gray image and synthetic video**  
## (C)key points detect##
Click the 'Open DLC' button on the 'Detection' tab will open the DLC GUI. Under this interface, please follow the steps below to operate:  
1. Click on the 'create Project' button on the homepage and input the information of "Project", "Experimenter" and "Location". Then check the "Copy videos to project folder"(If unchecked, the original video will be deleted).Finally, click the "Browse videos" button and select the directory "../your project name/Preprocessed/gray";  
2. Click 'Edit config.yaml', find the 'bodyparts' parameter and expand it. Change the number and name of 'bodyparts' according to the number of key points and parts you need to mark. Then click 'save'(**Please make sure to arrange the bodyparts in the topological order of the vertices of the polygon**);  
3. Click on the 'Extract Frames' button under the 'Extract Frames' tab to extract the frames used for labeling (if you are not satisfied with the automatically extracted frames, you can also choose the manual extraction method);  
4. Click on the 'Label Frames' button on the 'Label frames' tab and select the folder ../your DLC_directory/labeled data/<the unique directory>, this will open a GUI called napari, where key points can be marked sequentially. After all frames are marked, users can press 'ctrl+s' to save the marked results;  
5. Under the 'Create training dataset' tab, select a network structure based on your computer's configuration and needs, with 'Shuffle' set to 1. Then click on the 'Training Dataset' button;  
6. Modify parameters according to your own situation in 'Train network';  
7. Import file ../your project name/Preprocessed/gray/BCS.avi in the 'Analyze videos' tab and be sure to check 'Save result(s) as csv', then click on the 'Analyze Video' button and wait it finish;  
8. Exit DLC GUI.  
For more operations and details of DLC, please refer to the official documentation provided by [DLC](https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/). 
