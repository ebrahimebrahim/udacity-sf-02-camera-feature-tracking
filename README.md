# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />


The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.

---

Above is the original README for the assignment. Below are my (Ebrahim's) additions.

---

## How I was able to build on ubuntu

1. Install `sudo apt install libgtk2.0-dev pkg-config`
2. Get opencv and opencv\_contrib source code, and then build using the following cmake:
```
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib source directory>/modules <opencv source directory>
```
3. Use `sudo make install` to install opencv systemwide
4. Now you can cmake and make as usual for this project

## Executables

There are two executable build targets: `2D_feature_tracking` and `create_table`

`2D_feature_tracking` is the main project, and includes a visualization of a particular keypoint detector and descriptor setup. The idea is to change parameters in the code, rebuild this target, and observe the effects in the visualization.

`create_table` is used to write the table that analyzes all possible detector and descriptor combinations. Run like this:
```
./create_table > table.csv
```