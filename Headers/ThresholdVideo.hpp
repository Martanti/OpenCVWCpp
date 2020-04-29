#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include <time.h>

#pragma once

using cv::Mat;
using std::vector;
using cv::cuda::GpuMat;

class CThresholdVideo{
public:
    void Threshold();

    struct SRangeLimits{
        vector<unsigned char> vucLowerRangeLower;
        vector<unsigned char> vucLowerRangeUpper;
        vector<unsigned char> vucUpperRangeLower;
        vector<unsigned char> vucUpperRangeUpper;
    };

private:
    
    //cpu
    void ImageThreshold(Mat &rmatImage, SRangeLimits & srlLimits  , Mat &matOutput);

    void ApplyConvexHull(vector<Mat>& rvmatContours, vector<Mat>& rmatOutput);

    void GetCooridinatesOfBB(vector<Mat> vmetConvexHulls, vector<cv::Rect> & vrctBoundingRectangle);

    void GetEdges(Mat &rmatImage, Mat & rmatOutput);

    //GPU

    void ImageThreshold(GpuMat &rgmatImage, SRangeLimits & srlLimits  , GpuMat &gmatOutput);

    void ApplyConvexHull(vector<GpuMat>& rvgmatContours, vector<GpuMat>& rgmatOutput);

    void GetCooridinatesOfBB(vector<GpuMat> vmetConvexHulls, vector<cv::Rect> & vrctBoundingRectangle);

    void GetEdges(GpuMat &rgmatImage, GpuMat & rgmatOutput);
};

