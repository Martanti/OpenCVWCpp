#include <ThresholdVideo.hpp>

//what I use to compile this entire thing
//g++ -IHeaders -ISou#inclurce main.cpp Source/ThresholdVideo.cpp -o output `pkg-config --cflags --libs opencv4`
//g++ -d *
void CThresholdVideo::Threshold()
{

    cv::VideoCapture vcFootage; //= cv::VideoCapture(0); //this causes a warning
    Mat matVideoFrame;
    //GpuMat matVideoFrame;

    vcFootage.open("Source/TestVid.mp4");

    clock_t clckFrameBegining;
    while (true) //keep reading as long as there are frames
    {
        clckFrameBegining = clock();

        vcFootage.read(matVideoFrame);

        if (matVideoFrame.empty())
        {
            break;
        }

        Mat matFrameCopy;
        //GpuMat matFrameCopy;

        matVideoFrame.copyTo(matFrameCopy);

        SRangeLimits colours[3];

        SRangeLimits rlBlue;
        rlBlue.vucLowerRangeLower = {110, 190, 95};
        rlBlue.vucLowerRangeUpper = {130, 255, 200};
        rlBlue.vucUpperRangeLower = {50, 130, 135};
        rlBlue.vucUpperRangeUpper = {100, 100, 100};

        SRangeLimits rlRed;
        rlRed.vucLowerRangeLower = {0, 135, 135};
        rlRed.vucLowerRangeUpper = {15, 255, 255};
        rlRed.vucUpperRangeLower = {159, 135, 135};
        rlRed.vucUpperRangeUpper = {179, 255, 255};

        SRangeLimits rlYellow;
        rlYellow.vucLowerRangeLower = {137, 119, 0};
        rlYellow.vucLowerRangeUpper = {233, 214, 0};
        rlYellow.vucUpperRangeLower = {10, 135, 135};
        rlYellow.vucUpperRangeUpper = {100, 255, 255};

        Mat matMaskBlue;
        //GpuMat matMaskBlue;
        this->ImageThreshold(matFrameCopy, rlBlue, matMaskBlue);

        Mat matEdgesBlue;
        //GpuMat matEdgesBlue;
        this->GetEdges(matMaskBlue, matEdgesBlue);

        vector<Mat> vmatContoursBlue;
        //vector<GpuMat> vmatContoursBlue;
        cv::findContours(matEdgesBlue, vmatContoursBlue, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        vector<Mat> vmatConvexHullBlue;
        //vector<GpuMat> vmatConvexHullBlue;

        this->ApplyConvexHull(vmatContoursBlue, vmatConvexHullBlue);

        vector<cv::Rect> vrctBoundigBoxesBlue;
        this->GetCooridinatesOfBB(vmatConvexHullBlue, vrctBoundigBoxesBlue);

        for (auto item : vrctBoundigBoxesBlue)
        {
            cv::rectangle(matFrameCopy, item, {0, 255, 0}, 3);
        }

        cv::imshow("Cool video", matFrameCopy);
        std::cout << "Seconds for a frame: " << (float)(clock() - clckFrameBegining) / CLOCKS_PER_SEC << " \n";

        if (cv::waitKey(30) >= 0)
        {
            break;
        }
    }

    matVideoFrame.release();
    cv::destroyAllWindows();
}

void CThresholdVideo::ImageThreshold(Mat &rmatImage, SRangeLimits &srlLimits, Mat &matOutput)
{
    //Hue Saturation Value
    Mat matHSV;

    cv::cvtColor(rmatImage, matHSV, cv::COLOR_BGR2HSV);

    Mat matThresholdedLow, matThresholdedUpper;

    cv::inRange(matHSV, srlLimits.vucLowerRangeLower, srlLimits.vucLowerRangeUpper, matThresholdedLow);
    cv::inRange(matHSV, srlLimits.vucUpperRangeLower, srlLimits.vucUpperRangeUpper, matThresholdedUpper);

    cv::bitwise_or(matThresholdedLow, matThresholdedUpper, matOutput);
}

void CThresholdVideo::GetEdges(Mat &rmatImage, Mat &rmatOutput)
{

    Mat matMorphedImage;
    Mat matKernel = Mat(5, 5, CV_64F, 1);
    cv::morphologyEx(rmatImage, matMorphedImage, cv::MORPH_OPEN, matKernel);

    Mat matBlurredImage;
    cv::medianBlur(matMorphedImage, matBlurredImage, 5);

    cv::Canny(matBlurredImage, rmatOutput, 80, 160);
}

void CThresholdVideo::ApplyConvexHull(vector<Mat> &rvmatContours, vector<Mat> &rmatOutput)
{

    vector<Mat> vmatApproximatedContours;
    vmatApproximatedContours.reserve(2000);

    Mat matApproximatedPolygon;
    for (auto item : rvmatContours)
    {
        cv::approxPolyDP(item, matApproximatedPolygon, 10, true);
        vmatApproximatedContours.emplace_back(matApproximatedPolygon);
    }
    vmatApproximatedContours.shrink_to_fit();

    rmatOutput.reserve(20);
    for (auto item : vmatApproximatedContours)
    {
        Mat matConvexHulled;
        cv::convexHull(item, matConvexHulled);
        rmatOutput.emplace_back(matConvexHulled);
    }
    rmatOutput.shrink_to_fit();
}

void CThresholdVideo::GetCooridinatesOfBB(vector<Mat> vmetConvexHulls, vector<cv::Rect> &vrctBoundingRectangle)
{

    vector<Mat> vmatFilteredHulls;

    for (auto item : vmetConvexHulls)
    {
        unsigned char uiTotal = item.total();
        unsigned char uiUpperLimit = 5;
        unsigned char uiLowerLimit = 3;

        if (uiTotal >= uiLowerLimit && uiTotal <= uiUpperLimit)
        {
            vmatFilteredHulls.emplace_back(item);
        }
    }

    vrctBoundingRectangle.reserve(2000);

    for (auto item : vmatFilteredHulls)
    {
        cv::Rect rctBoundigBox = cv::boundingRect(item);
        vrctBoundingRectangle.emplace_back(rctBoundigBox);
    }
    vrctBoundingRectangle.shrink_to_fit();
}
//GPU overloads

void CThresholdVideo::ImageThreshold(GpuMat &rgmatImage, SRangeLimits &srlLimits, GpuMat &gmatOutput)
{
    //Hue Saturation Value
    GpuMat gmatHSV;

    cv::cvtColor(rgmatImage, gmatHSV, cv::COLOR_BGR2HSV);

    Mat matThresholdedLow, matThresholdedUpper;

    cv::inRange(gmatHSV, srlLimits.vucLowerRangeLower, srlLimits.vucLowerRangeUpper, matThresholdedLow);
    cv::inRange(gmatHSV, srlLimits.vucUpperRangeLower, srlLimits.vucUpperRangeUpper, matThresholdedUpper);

    cv::bitwise_or(matThresholdedLow, matThresholdedUpper, gmatOutput);

}   

void CThresholdVideo::ApplyConvexHull(vector<GpuMat> &rvgmatContours, vector<GpuMat> &rgmatOutput)
{

    vector<GpuMat> vgmatApproximatedContours;
    vgmatApproximatedContours.reserve(2000);

    Mat matApproximatedPolygon;
    for (auto item : rvgmatContours)
    {
        cv::approxPolyDP(item, matApproximatedPolygon, 10, true);
        vgmatApproximatedContours.emplace_back(matApproximatedPolygon);
    }
    vgmatApproximatedContours.shrink_to_fit();

    rgmatOutput.reserve(20);
    for (auto item : vgmatApproximatedContours)
    {
        Mat matConvexHulled;
        cv::convexHull(item, matConvexHulled);
        rgmatOutput.emplace_back(matConvexHulled);
    }
    rgmatOutput.shrink_to_fit();
}

void CThresholdVideo::GetCooridinatesOfBB(vector<GpuMat> vmetConvexHulls, vector<cv::Rect> &vrctBoundingRectangle)
{
    vector<GpuMat> vgmatFilteredHulls;

    for (auto item : vmetConvexHulls)
    {
        unsigned char uiTotal = item.elemSize();
        unsigned char uiUpperLimit = 5;
        unsigned char uiLowerLimit = 3;

        if (uiTotal >= uiLowerLimit && uiTotal <= uiUpperLimit)
        {
            vgmatFilteredHulls.emplace_back(item);
        }
    }

    vrctBoundingRectangle.reserve(2000);

    for (auto item : vgmatFilteredHulls)
    {
        cv::Rect rctBoundigBox = cv::boundingRect(item);
        vrctBoundingRectangle.emplace_back(rctBoundigBox);
    }
    vrctBoundingRectangle.shrink_to_fit();
}

void CThresholdVideo::GetEdges(GpuMat &rgmatImage, GpuMat &rgmatOutput)
{
    GpuMat gmatMorphedImage;
    GpuMat gmatKernel = GpuMat(5, 5, CV_64F, 1);
    cv::morphologyEx(rgmatImage, gmatMorphedImage, cv::MORPH_OPEN, gmatKernel);

    GpuMat gmatBlurredImage;
    cv::medianBlur(gmatMorphedImage, gmatBlurredImage, 5);

    cv::Canny(gmatBlurredImage, rgmatOutput, 80, 160);
}
