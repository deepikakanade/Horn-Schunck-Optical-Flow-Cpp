#ifndef opticalFlow_h
#define opticalFlow_h
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/video.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

void preprocess(cv::Mat imagePrevRaw, cv::Mat imageNextRaw, cv::Mat &imagePrev, cv::Mat &imageNext);
void plot1(cv::Mat img, cv::Mat u, cv::Mat v, int delta, float scale, std::string savePath);
void plot2(cv::Mat img_sz, cv::Mat velx, cv::Mat vely, float scale, std::string savePath);

void preprocess(cv::Mat imagePrevRaw, cv::Mat imageNextRaw, cv::Mat &imagePrev, cv::Mat &imageNext) {
    // Converting RGB to Gray if necessary
    if (imagePrevRaw.channels() > 1) {
        cv::cvtColor(imagePrevRaw, imagePrev, cv::COLOR_BGR2GRAY);
    }
    else {
        imagePrevRaw.copyTo(imagePrev);
    }
    
    if (imageNextRaw.channels() > 1) {
        cv::cvtColor(imageNextRaw, imageNext, cv::COLOR_BGR2GRAY);
    }
    else {
        imageNextRaw.copyTo(imageNext);
    }
}

void plot1(cv::Mat img, cv::Mat u, cv::Mat v, int delta, float scale, std::string savePath) {
    cv::Mat imgPlot = img.clone();
    
    for (int i = 0; i < img.rows; i += delta){
        for (int j = 0; j < img.cols; j += delta) {
            // Calculating u + du
            int iEnd = i + (int)(u.at<double>(i, j) * scale);
            // Calculating v + dv
            int jEnd = j + (int)(v.at<double>(i, j) * scale);
            
            //std::cout << i << " " << j << " " << u.at<double>(i, j) << " " << v.at<double>(i, j) << " " << iEnd << " " << jEnd << std::endl;
            cv::Point p = cv::Point(i, j);
            cv::Point p2 = cv::Point(iEnd, jEnd);
            
            cv::arrowedLine(imgPlot, p, p2, CV_RGB(0, 255, 0), 1, cv::LINE_AA);
        }
    }
    
    cv::namedWindow("Plot1 Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Plot1 Image", imgPlot);
    cv::imwrite(savePath+"Plot1.png", imgPlot);
}

void plot2(cv::Mat img_sz, cv::Mat velx, cv::Mat vely, float scale, std::string savePath) {
    cv::Mat imgPlot = img_sz.clone();
    double l_max = -10;
    
    for (int y = 0; y < img_sz.rows; y += 10) {
        for (int x = 0; x < img_sz.cols; x += 10) {
            double dx = velx.at<double>(y, x) * scale;
            double dy = vely.at<double>(y, x) * scale;
            double l = sqrt(dx*dx + dy*dy);
            if(l>l_max)
                l_max = l;
        }
    }
    
    for (int y = 0; y < img_sz.rows; y += 10){
        for (int x = 0; x < img_sz.cols; x += 10){
            double dx = velx.at<double>(y, x) * scale;
            double dy = vely.at<double>(y, x) * scale;
            cv::Point p = cv::Point(x, y);
            double l = sqrt(dx*dx + dy*dy);
            if (l > 0){
                double spinSize = 5.0 * l/l_max;
                cv::Point p2 = cv::Point(p.x + (int)(dx), p.y + (int)(dy));
                cv::line(imgPlot, p, p2, CV_RGB(0, 255, 0), 1, cv::LINE_AA);

                double angle;
                angle = atan2( (double) p.y - p2.y, (double) p.x - p2.x );

                p.x = (int) (p2.x + spinSize * cos(angle + 3.1416 / 4));
                p.y = (int) (p2.y + spinSize * sin(angle + 3.1416 / 4));
                cv::line(imgPlot, p, p2, CV_RGB(0,255,0), 1, cv::LINE_AA, 0);

                p.x = (int) (p2.x + spinSize * cos(angle - 3.1416 / 4));
                p.y = (int) (p2.y + spinSize * sin(angle - 3.1416 / 4));
                cv::line(imgPlot, p, p2, CV_RGB(0,255,0), 1, cv::LINE_AA, 0);
            }
        }
    }
    
    cv::namedWindow("Plot2 Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Plot2 Image", imgPlot);
    cv::imwrite(savePath + "Plot2.png", imgPlot);
}

#endif /* opticalFlow_h */
