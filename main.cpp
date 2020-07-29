#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

void preprocess(cv::Mat imagePrevRaw, cv::Mat imageNextRaw, cv::Mat &imagePrev, cv::Mat &imageNext) {
        
    // Converting RGB to Gray if necessary
    if (imagePrevRaw.channels() > 1){
        cv::cvtColor(imagePrevRaw, imagePrev, cv::COLOR_BGR2GRAY);
    }
    else{
        imagePrevRaw.copyTo(imagePrev);
    }
    if (imageNextRaw.channels() > 1){
        cv::cvtColor(imageNextRaw, imageNext, cv::COLOR_BGR2GRAY);
    }
    else{
        imageNextRaw.copyTo(imageNext);
    }
}

void getGradients(cv::Mat imagePrev, cv::Mat imageNext, cv::Mat &gradX, cv::Mat &gradY, cv::Mat &gradT){
    cv::Mat imagePrevNorm, imageNextNorm;
    
    // Convert image values to 64 bit float single channel
    imagePrev.convertTo(imagePrevNorm, CV_64FC1);
    imageNext.convertTo(imageNextNorm, CV_64FC1);
    
    // Obtain gradient in X direction and Y direction using a Sobel Filter
    Sobel(imagePrevNorm, gradX, -1, 1, 0, 3);
    Sobel(imagePrevNorm, gradY, -1, 0, 1, 3);
    
    // Obtain gradient in T direction by subtracting both the images
    gradT = imageNextNorm - imagePrevNorm;
    
}

void hornSchunckFlow(cv::Mat imagePrev, cv::Mat imageNext, cv::Mat &u, cv::Mat &v){
    // Get gradients
    cv::Mat gradX, gradY, gradT;
    getGradients(imagePrev, imageNext, gradX, gradY, gradT);
    
    std::cout << "Gradient X Size: " << gradX.size() << std::endl;
    std::cout << "Gradient Y Size: " << gradY.size() << std::endl;
    std::cout << "Gradient T Size: " << gradT.size() << std::endl;
    
    // Initialize u and v matrices with zeros of same size and format as gradT
    u = cv::Mat::zeros(gradT.rows, gradT.cols, CV_64FC1);
    v = cv::Mat::zeros(gradT.rows, gradT.cols, CV_64FC1);
       
    int windowSize = 5; //TBD to generalize
    int maxIterations = 100; //TBD to generalize
    double alpha = 1; //TBD to generalize
    
    // Get kernel and anchor for averaging u and v matrices
    cv::Mat kernel = cv::Mat::ones(windowSize, windowSize, CV_64FC1) / pow(windowSize, 2);
    cv::Point anchor(windowSize-(windowSize/2)-1,windowSize-(windowSize/2)-1);
    
    for (int i = 0; i < maxIterations; i++) {
        cv::Mat uAvg, vAvg, gradXuAvg, gradYvAvg, gradXgradX, gradYgradY, updateConst, uUpdateConst, vUpdateConst; //TBD to generalize
        
        filter2D(u, uAvg, u.depth(), kernel, anchor, 0, cv::BORDER_CONSTANT);
        filter2D(v, vAvg, v.depth(), kernel, anchor, 0, cv::BORDER_CONSTANT);
        
        multiply(gradX, uAvg, gradXuAvg);
        multiply(gradY, vAvg, gradYvAvg);
        multiply(gradX, gradX, gradXgradX);
        multiply(gradY, gradY, gradYgradY);
        
        divide((gradXuAvg + gradYvAvg + gradT), (pow(alpha,2) + gradXgradX +gradYgradY), updateConst);
        multiply(gradX, updateConst, uUpdateConst);
        multiply(gradY, updateConst, vUpdateConst);
        
        u = uAvg - uUpdateConst;
        v = vAvg - vUpdateConst;
    }
}

void plot1(cv::Mat img, cv::Mat u, cv::Mat v, int delta, float scale){
    cv::Mat imgPlot = img.clone();
    for (int i=0; i<img.rows; i+=delta){
        for (int j=0; j<img.cols; j+=delta){
            int iEnd = i + (int)(u.at<double>(i, j) * scale);
            int jEnd = j + (int)(v.at<double>(i, j) * scale);
            //std::cout << i << " " << j << " " << u.at<double>(i, j) << " " << v.at<double>(i, j) << " " << iEnd << " " << jEnd << std::endl;
            cv::Point p = cv::Point(i, j);
            cv::Point p2 = cv::Point(iEnd, jEnd);
            cv::arrowedLine(imgPlot, p, p2, CV_RGB(0, 255, 0), 1, cv::LINE_AA);
        }
    }
    cv::namedWindow("Plot1 Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Plot1 Image", imgPlot);
}

void plot2(cv::Mat img_sz, cv::Mat velx, cv::Mat vely, float scale){
    cv::Mat imgPlot = img_sz.clone();
    double l_max = -10;
    for (int y = 0; y < img_sz.rows; y+=10){
        for (int x = 0; x < img_sz.cols; x+=10){
            double dx = velx.at<double>(y, x) * scale;
            double dy = vely.at<double>(y, x) * scale;
            double l = sqrt(dx*dx + dy*dy);
            if(l>l_max) l_max = l;
        }
    }
    for (int y = 0; y < img_sz.rows; y+=10){
        for (int x = 0; x < img_sz.cols; x+=10){
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
}


int main(int argc, char* argv[]) {
    
    // Read input images
    //cv::Mat imagePrevRaw = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    //cv::Mat imageNextRaw = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
    
    cv::Mat imagePrevRaw = cv::imread(argv[1]);
    cv::Mat imageNextRaw = cv::imread(argv[2]);
    
    // Check if files are loaded properly
    if (!(imagePrevRaw.data) || !(imageNextRaw.data)) {
        std::cout << "Can't read the images. Please check the path." << std::endl;
        return -1;
    }

    // Check if image sizes are same
    if ( imagePrevRaw.size() != imageNextRaw.size() ) {
        std::cout << "Image sizes are different. Please provide images of same size." << std::endl;
        return -1;
    }
    
    std::cout << "Raw Previous Image Size: " << imagePrevRaw.size() << std::endl;
    std::cout << "Raw Previous Image Channel Size: " << imagePrevRaw.channels() << std::endl;
    std::cout << "Raw Next Image Size: " << imageNextRaw.size() << std::endl;
    std::cout << "Raw Next Image Channel Size: " << imageNextRaw.channels() << std::endl;
    
    cv::Mat imagePrev, imageNext;
    preprocess(imagePrevRaw, imageNextRaw, imagePrev, imageNext);
    
    std::cout << "Previous Image Size: " << imagePrev.size() << std::endl;
    std::cout << "Previous Image Channel Size: " << imagePrev.channels() << std::endl;
    std::cout << "Next Image Size: " << imageNext.size() << std::endl;
    std::cout << "Next Image Channel Size: " << imageNext.channels() << std::endl;
    
    
    cv::Mat u, v;
    hornSchunckFlow(imagePrev, imageNext, u, v);
    std::cout << "u Size: " << u.size() << std::endl;
    std::cout << "v Size: " << v.size() << std::endl;
    
    cv::FileStorage file_1(argv[3], cv::FileStorage::WRITE);
    file_1 << "u matrix" << u;
    cv::FileStorage file_2(argv[4], cv::FileStorage::WRITE);
    file_2 << "v matrix" << v;
    
    plot1(imagePrevRaw, u, v, 10, 20);
    plot2(imagePrevRaw, u, v, 20);
    
    cv::waitKey(0);
    return 0;
    
}
