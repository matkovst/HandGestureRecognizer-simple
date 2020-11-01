#pragma once

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "skin_segm.h"
#include "bg_subtr.h"

#define DEFAULT_ALPHA 1/(25.0*60)
#define DEFAULT_SKIN_PRIOR 70/100.f
#define MORPH_KSIZE 3
#define DEFAULT_THRESH 20
#define SKIN_THRESH 0.2f
#define MIN_SKIN_DECISION 0.05f
#define WINDOW_NAME "Gesture recognition"

class GestureRecognizer
{
public:
    GestureRecognizer(const double bgAlpha = DEFAULT_ALPHA, const float skinPrior = DEFAULT_SKIN_PRIOR);
    ~GestureRecognizer();
    int recognize(const cv::Mat& _frame, bool visualize = false);
    bool seeHand();

private:
    cv::Ptr<cv::BackgroundSubtractor> m_bgSubtractor;
    std::shared_ptr<SkinSegmentator> m_skinSegmentator;
    cv::Ptr<cv::DISOpticalFlow> m_DISOptFlow;
    double m_bgAlpha { DEFAULT_ALPHA };
    bool m_hand { false };
    cv::Mat m_foregroundMask;
    std::vector<cv::Point> m_handContour;
    std::vector<cv::Point> m_handHull;
    std::vector<cv::Point2f> m_fingerLandmarks;
    cv::Point2f m_palmCenter;
    cv::Rect m_fingerBox;
    cv::Mat m_flow;
    cv::Mat m_prevFrame;
    cv::Mat m_prevColorizedFg;

    bool observeHand(const cv::Mat& frame, const float skinThresh = SKIN_THRESH, const float decisionThresh = MIN_SKIN_DECISION);
    void generateLandmarks();
    void getLandmarks(std::vector<cv::Point2f>& out);
    void getPalmCenter();
    void trackLandmarks(const cv::Mat& prevFrame, const cv::Mat& frame);
    void releaseLandmarks();
    cv::Rect handROI(const int W, const int H);
    cv::Point getROIOffset(const int W, const int H);
    int getMaxAreaContourId(std::vector<std::vector<cv::Point>> contours);

    void drawGestureArea(cv::Mat& frame);
    void drawMotionField(const cv::Mat& flow, cv::Mat& out, int stride);
    void drawHeatmap(const cv::Mat& flow, cv::Mat& out);
    cv::Scalar jet(cv::Point2f vv);
};