/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"


// Writes a row for each frame, given a choice of detector and a choice of descriptor
// detectors: SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
// descriptors: BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
void write_rows(std::ostream & os, std::string detectorType, std::string descriptorType) {

    // data location
    std::string dataPath = "../";

    // camera
    std::string imgBasePath = dataPath + "images/";
    std::string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    std::string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    size_t dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    Ringbuf<DataFrame> dataBuffer{dataBufferSize}; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        std::ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgStartIndex + imgIndex;
        std::string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);

        /* DETECT IMAGE KEYPOINTS */
        double t0 = get_ticks_ms();

        // extract 2D keypoints from current image
        std::vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        if (detectorType.compare("SHITOMASI") == 0)
            detKeypointsShiTomasi(keypoints, imgGray, false,true);
        else if (detectorType.compare("HARRIS") == 0)
            detKeypointsHarris(keypoints, imgGray, false, true);
        else
            detKeypointsModern(keypoints, imgGray, detectorType, false, true);

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            for (auto it = keypoints.begin(); it!=keypoints.end();) {
                if (! vehicleRect.contains(it->pt))
                    it = keypoints.erase(it);
                else
                    ++it;
            }
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = std::move(keypoints);

        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        std::string descriptorType = "BRIEF"; // BRIEF, ORB, FREAK, AKAZE, SIFT
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType,true);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = std::move(descriptors);


        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            std::vector<cv::DMatch> matches;
            std::string matcherType = "MAT_FLANN";        // MAT_BF, MAT_FLANN
            std::string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            std::string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN


            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);


            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = std::move(matches);


            /* WRITE DATA */

            // time in ms to do the keypoint detection, descriptor extraction, and matching
            os << get_ticks_ms()-t0;

            

            os << "\n";





            // visualize matches between current and previous image
            bVis = true; // (Turn off when generating rows of table)
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                (dataBuffer.end() - 1)->kptMatches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                std::string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                std::cout << "Press key to continue to next image" << std::endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // end of loop over all images
}


int main() {

    write_rows(std::cout, "SHITOMASI", "BRISK");

    return 0;
}