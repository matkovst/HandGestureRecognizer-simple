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
#define DEFAULT_THRESH 20
#define DEFAULT_HAND_FRAMES 15
#define MORPH_KSIZE 3
#define SKIN_THRESH 0.2f
#define MIN_SKIN_DECISION 0.05f

enum GESTURE_TYPE
{
    NONE            = 0,
    ACT_LFINGER     = 1,
    ACT_RFINGER     = 2,
    ACT_LRFINGER    = 3,
    ACT_MFINGER     = 4,
    ACT_LMFINGER    = 5,
    ACT_RMFINGER    = 6,
    ACT_LRMFINGER   = 7,
    ACT_IFINGER     = 8,
    ACT_LIFINGER    = 9,
    ACT_RIFINGER    = 10,
    ACT_LRIFINGER   = 11,
    ACT_MIFINGER    = 12,
    ACT_LMIFINGER   = 13,
    ACT_RMIFINGER   = 14,
    ACT_LRMIFINGER  = 15,
    ACT_TFINGER     = 16,
    ACT_LTFINGER    = 17,
    ACT_RTFINGER    = 18,
    ACT_LRTFINGER   = 19,
    ACT_MTFINGER    = 20,
    ACT_LMTFINGER   = 21,
    ACT_RMTFINGER   = 22,
    ACT_LRMTFINGER  = 23,
    ACT_ITFINGER    = 24,
    ACT_LITFINGER   = 25,
    ACT_RITFINGER   = 26,
    ACT_LRITFINGER  = 27,
    ACT_MITFINGER   = 28,
    ACT_LMITFINGER  = 29,
    ACT_RMITFINGER  = 30,
    ACT_LRMITFINGER = 31
};

static const char* fingerNames[] = { "little", "ring", "middle", "index", "thumb" };

struct PalmString
{
    cv::Point start;
    cv::Point end;
    float initialLength { 0.f };

    PalmString(cv::Point _start, cv::Point _end)
    {
        start = _start;
        end = _end;
        initialLength = length();
    }
    float length()
    {
        return sqrtf(powf(end.x - start.x, 2) + powf(end.y - start.y, 2));
    }
    bool hasAct()
    {
        return length() < initialLength*(3/4.f);
    }
};

class GestureRecognizer
{
public:
    GestureRecognizer(const double bgAlpha = DEFAULT_ALPHA, const float skinPrior = DEFAULT_SKIN_PRIOR);
    ~GestureRecognizer();
    GESTURE_TYPE recognize(cv::Mat& _frame, bool debug = false, cv::Mat& debugFrame = cv::Mat());
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
    std::vector<PalmString> m_palmStrings;
    cv::Mat m_flow;
    cv::Mat m_prevFrame;
    cv::Mat m_prevColorizedFg;

    bool observeHand(const cv::Mat& frame, const float skinThresh = SKIN_THRESH, const float decisionThresh = MIN_SKIN_DECISION);
    void generateLandmarks();
    void getLandmarks(std::vector<cv::Point2f>& out);
    void getPalmCenter();
    void getPalmStrings();
    void trackLandmarks(const cv::Mat& prevFrame, const cv::Mat& frame);
    void releaseLandmarks();
    GESTURE_TYPE detectGesture();
    cv::Rect handROI(const int W, const int H);
    cv::Point getROIOffset(const int W, const int H);
    int getMaxAreaContourId(const std::vector<std::vector<cv::Point>>& contours);

    void drawGestureArea(cv::Mat& frame);
    void drawStrings(cv::Mat& frame);
    void drawMotionField(const cv::Mat& flow, cv::Mat& out, int stride);
    void drawHeatmap(const cv::Mat& flow, cv::Mat& out);
    cv::Scalar jet(cv::Point2f vv);
};