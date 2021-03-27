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


// bins for distribution of keypoint sizes
const std::vector<float> kp_size_bins = {4,10,40,70,100,130};



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
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, true);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = std::move(descriptors);


        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            std::vector<cv::DMatch> matches;
            std::string matcherType = "MAT_FLANN";        // MAT_BF, MAT_FLANN
            std::string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN


            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, "DES_BINARY", matcherType, selectorType);


            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = std::move(matches);


            /* WRITE DATA */

            std::string sep = ", "; // seperator

            // detector and descriptor names, then frame index
            os << detectorType << sep;
            os << descriptorType << sep;
            os << imgIndex << sep;

            // time in ms to do the keypoint detection, descriptor extraction, and matching
            os << get_ticks_ms()-t0 << sep;

            // number of keypoints found in this frame
            os << (dataBuffer.end()-1)->keypoints.size() << sep;

            // number of matches in this frame
            os << (dataBuffer.end() - 1)->kptMatches.size() << sep;

            // bins for distribution of keypoint sizes
            std::vector<float> kp_sizes;
            for (const auto & kp : (dataBuffer.end()-1)->keypoints)
                kp_sizes.push_back(kp.size);
            
            for (int i = 0; i < kp_size_bins.size()-1; ++i) {
                int count = 0;
                for (float s : kp_sizes){
                    if ((s >= kp_size_bins[i]) && (s < kp_size_bins[i+1]))
                        ++count;
                }
                os << count << sep;
            }
            
            // end of row of table
            os << "\n";



            // visualize matches between current and previous image
            bVis = false; // (Turn off when generating rows of table)
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

    std::ostream & output_target = std::cout;

    // write header
    output_target << "detector, descriptor, frame index, time (ms), keypoints found, matches with previous frame, ";
    for (int i = 0; i < kp_size_bins.size()-1; ++i) {
        output_target << "keypoint size bin " << kp_size_bins[i] << " to " << kp_size_bins[i+1] <<  ", ";
    }
    output_target << "\n";

    std::vector<std::string> detectors = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB"}; // TODO put SIFT back in, but don't use with ORB descriptor
    std::vector<std::string> descriptors = {"BRISK", "BRIEF", "ORB", "FREAK"}; // TODO put SIFT back in

    for (auto & detector : detectors)
        for (auto descriptor : descriptors)
            write_rows(output_target, detector, descriptor);
    
    // It seems that the AKAZE descriptor pretty much only works with AKAZE keypoints.
    // See https://github.com/kyamagu/mexopencv/issues/351
    write_rows(output_target, "AKAZE", "AKAZE");

    return 0;
}