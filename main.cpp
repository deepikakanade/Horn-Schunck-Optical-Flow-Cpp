#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

void preprocess_image(cv::Mat image1, cv::Mat image2, cv::Mat &Ix, cv::Mat &Iy) {
    
    cv::Mat grayImage1, grayImage2;
    
    // Converting RGB to Gray (OpenCV reads images in BGR format)
    cv::cvtColor(image1, grayImage1, CV_BGR2GRAY);
    cv::cvtColor(image2, grayImage2, CV_BGR2GRAY);
    
    cv::namedWindow("Gray Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Gray Image", grayImage1);
    
    cv::Mat grayImageNorm1;
    
    // convert Uint8 to double, Normalize between 0 - 1
    grayImage1.convertTo(grayImageNorm1, CV_64FC1, 1.0 / 255.0);

    // To have destination image of same depth as the source
    int ddepth = -1;

    // Find derivative of image intensity values in X direction with 3x3 kernel size
    Sobel(grayImage1, Ix, ddepth, 1, 0, 3);
    
    // Find derivative of image intensity values in Y direction with 3x3 kernel size
    Sobel(grayImage1, Iy, ddepth, 0, 1, 3);
    
}

int main(int argc, char* argv[]) {
    
	// Read input images
    cv::Mat image1 = cv::imread(argv[1]);
    cv::Mat image2 = cv::imread(argv[2]);

	// Check if files are loaded properly
    if (!(image1.data) || !(image2.data)) {
        std::cout << "Can't read the images. Please check the path." << std::endl;
		return -1;
    }

    // Check if image sizes are same
    if ( image1.size() != image2.size() ) {
        std::cout << "Size of the images is different. Please provide images of same size." << std::endl;
        return -1;
    }
    
    std::cout << "Shape of the image is: " << "[" << image1.size().height << ", " << image1.size().width << "]" << std::endl;
    std::cout << "Number of channels in the image is: " << image1.channels() << std::endl;
    
    namedWindow("Original Image 1", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original Image 1", image1);
    
    namedWindow("Original Image 2", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original Image 2", image2);
    
    cv::Mat Ix, Iy;
    
    // Conversion of BGR to Grayscale and edge detection
    preprocess_image(image1, image2, Ix, Iy);
    
    /*for(int row = 0; row < Iy.rows; row++) {
        for(int col = 0; col < Iy.cols; col++) {
            std::cout << "Ix is: " << Ix.at<double>(row,col) << " " << "Iy is: " << Iy.at<double>(row,col) << std::endl;
            std::cout << "Row: " << row << "Col: " << col << std::endl;
        }
    }*/
    
    cv::namedWindow("Ix Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Ix Image", Ix);
    
    cv::namedWindow("Iy Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Iy Image", Iy);
    
    cv::waitKey(0);
    
    return 0;
}
