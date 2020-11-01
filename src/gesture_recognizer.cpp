#pragma once

#include "gesture_recognizer.h"

GestureRecognizer::GestureRecognizer(const float bg_alpha, const float skin_prior)
{
    m_bgSubtractor = std::make_shared<SimpleBackgroundSubtractor>(bg_alpha);
    m_skinSegmentator = std::make_shared<SkinSegmentator>(skin_prior);
    m_DISOptFlow = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
}

GestureRecognizer::~GestureRecognizer() { }

bool GestureRecognizer::seeHand()
{
    return m_hand;
}

int GestureRecognizer::recognize(const cv::Mat& _frame, bool visualize)
{
    CV_Assert(!_frame.empty());

    cv::Mat frame = _frame.clone();
    const int H = frame.rows;
    const int W = frame.cols;
    if (m_prevFrame.empty())
    {
        m_prevFrame = frame.clone();
    }

    cv::GaussianBlur(frame, frame, cv::Size(3, 3), 0);

    float decisionThresh = (m_hand) ? 0.0001f : 0.01f;
    m_hand = observeHand(frame, 0.2f, decisionThresh);
    if (m_hand)
    {
        generateLandmarks();

        cv::Mat colorizedFg;
        cv::bitwise_and(frame, frame, colorizedFg, m_foregroundMask);
        if (m_prevColorizedFg.empty())
        {
            m_prevColorizedFg = colorizedFg.clone();
        }
        trackLandmarks(m_prevColorizedFg, colorizedFg);
        cv::swap(m_prevColorizedFg, colorizedFg);
    }
    else
    {
        releaseLandmarks();
    }

    cv::swap(m_prevFrame, frame);
    
    /* Display */
    if (visualize)
    {
        cv::Mat display = frame.clone();
        cv::Mat handDisplay = cv::Mat::zeros(H, W, CV_8UC3);
        cv::Mat fgDisplay;
        cv::cvtColor(m_foregroundMask, fgDisplay, cv::COLOR_GRAY2BGR);

        if (m_fingerLandmarks.size() == 5)
        {
            for (int i = 0; i < 5; i++)
            {
                cv::circle(display, m_fingerLandmarks[i], 25, cv::Scalar(40, 220, 40), 3);
                cv::line(display, m_fingerLandmarks[i], m_palmCenter, cv::Scalar(255, 255, 255), 2);
            }
            cv::circle(display, m_palmCenter, 15, cv::Scalar(255, 0, 255), 3);

            if (!m_handContour.empty()) cv::fillPoly(handDisplay, std::vector<std::vector<cv::Point>>(1, m_handContour), cv::Scalar(255, 255, 255));
            if (!m_handHull.empty()) cv::drawContours(fgDisplay, std::vector<std::vector<cv::Point>>(1, m_handHull), -1, cv::Scalar(255, 0, 255), 2);

            // optflow visualization
            if (!m_flow.empty())
            {
                cv::Mat flowDisplay;
                drawMotionField(m_flow, flowDisplay, 20);
                cv::imshow("Flow", flowDisplay);
            }
        }
        
        // if (!handHull.empty())
        // {
            // cv::drawContours(display, std::vector<std::vector<cv::Point>>(1, handHull), -1, cv::Scalar(255, 0, 255), 2);
            // for (int i = 0; i < 5; i++)
            // {
            //     cv::circle(display, goodFeatures[i], 25, cv::Scalar(0, 255, 0), 2);
            // }
            // std::vector<cv::Vec4i>::iterator it = handHullDefects.begin();
            // while( it != handHullDefects.end() ) {
            //     cv::Vec4i& v = (*it);
            //     int startidx = v[0]; cv::Point ptStart( handContour[startidx] );
            //     int endidx = v[1]; cv::Point ptEnd( handContour[endidx] );
            //     int faridx = v[2]; cv::Point ptFar( handContour[faridx] );
            //     float depth = v[3] / 256;

            //     cv::line( display, ptStart, ptEnd, cv::Scalar(0, 255, 0), 1 );
            //     // cv::line( display, ptStart, ptFar, cv::Scalar(0, 255, 0), 1 );
            //     // cv::line( display, ptEnd, ptFar, cv::Scalar(0, 255, 0), 1 );
            //     //cv::circle( display, ptFar, 4, cv::Scalar(0, 255, 0), 2 );
            //     it++;
            // }
        //     cv::fillPoly(handDisplay, std::vector<std::vector<cv::Point>>(1, handContour), cv::Scalar(255, 255, 255));
        // }

        drawGestureArea(display);
        cv::Mat stacked(display.rows, display.cols * 2, CV_8UC3, cv::Scalar::all(0));
        display.copyTo(stacked.colRange(0, display.cols));
        fgDisplay.copyTo(stacked.colRange(display.cols, display.cols * 2));
        cv::imshow(WINDOW_NAME, stacked);
    }

    return 0;
}

void GestureRecognizer::generateLandmarks()
{
    if (!m_fingerLandmarks.empty()) return;

    getLandmarks(m_fingerLandmarks);
    if (m_fingerLandmarks.empty()) return;
    getPalmCenter();

    double fingerArea = m_fingerBox.area();
    double handArea = cv::contourArea(m_handContour);
    bool passArea = (fingerArea < handArea) ? true : false;

    bool passHull = (m_handHull.size() == 7) ? true : false;

    if (!passArea || !passHull) m_fingerLandmarks.clear();
}

void GestureRecognizer::getLandmarks(std::vector<cv::Point2f>& out)
{
    const int H = m_foregroundMask.rows;
    const int W = m_foregroundMask.cols;
    cv::Mat morphed_fgmask_ROI = m_foregroundMask(handROI(W, H));
    std::vector<cv::Point> goodFeatures(5);
    std::vector<int> handHull_int;
    std::vector<cv::Vec4i> handHullDefects;
    if (m_handContour.size() > 0)
    {
            cv::convexHull(m_handContour, handHull_int, false, false);
            cv::convexityDefects(m_handContour, handHull_int, handHullDefects);

            // Get top 5 points
            std::vector<cv::Point> handHullSorted(m_handHull);
            std::sort(handHullSorted.begin(), handHullSorted.end(),
                [](const cv::Point &a, const cv::Point &b) -> bool const { return (a.y) < (b.y); });
            for (int i = 0; i < 5; i++) out.push_back(cv::Point2f((float)handHullSorted[i].x, (float)handHullSorted[i].y));
    }
}

void GestureRecognizer::getPalmCenter()
{
    m_fingerBox = cv::boundingRect(m_fingerLandmarks);

    float handCenter_x = m_fingerBox.x + m_fingerBox.width/2;
    float handCenter_y = m_fingerBox.y + m_fingerBox.height*1.2;
    m_palmCenter = cv::Point2f(handCenter_x, handCenter_y);
}

void GestureRecognizer::trackLandmarks(const cv::Mat& prevFrame, const cv::Mat& frame)
{
    if (m_fingerLandmarks.empty()) return;

    cv::Mat prevGray, gray;
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    /* track with LK optical flow */
    std::vector<uchar> status;
    std::vector<float> err;
    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 20, 0.01);
    std::vector<cv::Point2f> currFingerLandmarks;
    cv::calcOpticalFlowPyrLK(prevGray, gray, m_fingerLandmarks, currFingerLandmarks, status, err, cv::Size(35, 35), 5, criteria);

    /* Update landmarks */
    std::swap(currFingerLandmarks, m_fingerLandmarks);
    getPalmCenter();

    // m_fingerLandmarks.clear();
    // getLandmarks(m_fingerLandmarks);
    // m_DISOptFlow->calc(prevGray, gray, m_flow);
    // std::swap(prevGray, gray);
}

void GestureRecognizer::releaseLandmarks()
{
    m_fingerLandmarks.clear();
}

bool GestureRecognizer::observeHand(const cv::Mat& frame, const float skin_thresh, const float decisionThresh)
{
    static int seeHand = 0;
    m_handContour.clear();
    m_handHull.clear();

    /* Derive foreground mask */
    const int H = frame.rows;
    const int W = frame.cols;
    cv::Mat fgmask;
    m_bgSubtractor->apply(frame, fgmask, (!m_hand));
    cv::threshold(fgmask, fgmask, 70, 255, cv::THRESH_BINARY);

    cv::Mat open_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(MORPH_KSIZE, MORPH_KSIZE));
    cv::morphologyEx(fgmask, m_foregroundMask, cv::MORPH_OPEN, open_kernel, cv::Point(-1, -1), 1);
    cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(MORPH_KSIZE, MORPH_KSIZE));
    cv::morphologyEx(m_foregroundMask, m_foregroundMask, cv::MORPH_CLOSE, close_kernel, cv::Point(-1, -1), 1);

    cv::Mat colorized_fgmask;
    cv::bitwise_and(frame, frame, colorized_fgmask, m_foregroundMask);
    cv::Mat RoI = colorized_fgmask(handROI(W, H));

    /* Derive largest contour = hand */
    cv::Mat morphed_fgmask_ROI = m_foregroundMask(handROI(W, H));
    std::vector<std::vector<cv::Point>> contours;
    int handContourId = -1;
    cv::findContours( morphed_fgmask_ROI, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, getROIOffset(W, H) );
    if (contours.empty())
    {
        seeHand--;
        if (seeHand < 0) seeHand = 0;
        return (seeHand >= 6);
    }
    handContourId = getMaxAreaContourId(contours);
    m_handContour = contours.at(handContourId);
    double eps = 0.001*cv::arcLength(m_handContour, true);
    cv::approxPolyDP(m_handContour, m_handContour, eps, true);

    /* Derive hand convex hull */
    std::vector<cv::Vec4i> handHullDefects;
    cv::convexHull(m_handContour, m_handHull, false, false);
    eps = 0.01*cv::arcLength(m_handHull, true);
    cv::approxPolyDP(m_handHull, m_handHull, eps, true);

    /* Estimate skin */
    cv::Mat skin_mask;
    m_skinSegmentator->segment_skin(RoI, skin_mask, RoI);
    cv::threshold(skin_mask, skin_mask, skin_thresh, 1.f, cv::THRESH_BINARY);
    float total_skin_sum = (float)cv::sum(skin_mask)[0];
    float normed_sum = total_skin_sum / (RoI.rows * RoI.cols);

    seeHand = (normed_sum >= decisionThresh && normed_sum < 0.8) ? (seeHand + 1) : (seeHand - 1);
    if (seeHand < 0) seeHand = 0;
    if (seeHand > 12) seeHand = 12;

    return (seeHand >= 6);
}

void GestureRecognizer::drawGestureArea(cv::Mat& frame)
{
    cv::Rect RoI = handROI(frame.cols, frame.rows);
    if (m_hand) 
    {
        cv::rectangle(frame, RoI, cv::Scalar(40, 220, 40), 2);
        cv::putText(frame, "Hand", cv::Point(RoI.x + 10, RoI.y + 20), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 255, 0), 2, cv::LINE_4);
    }
    else 
    {
        cv::rectangle(frame, RoI, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, "No hand", cv::Point(RoI.x + 10, RoI.y + 20), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 255), 2, cv::LINE_4);
    }
}

int GestureRecognizer::getMaxAreaContourId(std::vector<std::vector<cv::Point>> contours)
{
    if (contours.empty()) return -1;

    double maxArea = 0;
    int maxAreaContourId = -1;
    for (int i = 0; i < contours.size(); i++)
    {
        double newArea = cv::contourArea(contours.at(i));
        if (newArea > maxArea)
        {
            maxArea = newArea;
            maxAreaContourId = i;
        }
    }
    return maxAreaContourId;
}

cv::Rect GestureRecognizer::handROI(const int W, const int H)
{
    int x1 = (int)(0.01 * W);
    int y1 = (int)(0.01 * H);
    int x2 = (int)(0.51 * W);
    int y2 = (int)(0.99 * H);
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}

cv::Point GestureRecognizer::getROIOffset(const int W, const int H)
{
    return cv::Point((int)(0.01 * W), (int)(0.01 * H));
}



// Convert vector magnitude to jet color
cv::Scalar GestureRecognizer::jet(cv::Point2f vv)
{
    double magn = cv::norm(vv);
    unsigned int H = 255 - (255 - magn) * 280 / 255;
    unsigned int hi = (H/60) % 6;
    float S=1.f;
    float V=1.f;
    float f = H/60.f - H/60;
    float p = V * (1 - S);
    float q = V * (1 - f * S);
    float t = V * (1 - (1 - f) * S);
    cv::Point3f res;
    if( hi == 0 ) //R = V,  G = t,  B = p
        res = cv::Point3f( p, t, V );
    if( hi == 1 ) // R = q, G = V,  B = p
        res = cv::Point3f( p, V, q );
    if( hi == 2 ) // R = p, G = V,  B = t
        res = cv::Point3f( t, V, p );
    if( hi == 3 ) // R = p, G = q,  B = V
        res = cv::Point3f( V, q, p );
    if( hi == 4 ) // R = t, G = p,  B = V
        res = cv::Point3f( V, p, t );
    if( hi == 5 ) // R = V, G = p,  B = q
        res = cv::Point3f( q, p, V );
    int b = int(std::max(0.f, std::min (res.x, 1.f)) * 255.f);
    int g = int(std::max(0.f, std::min (res.y, 1.f)) * 255.f);
    int r = int(std::max(0.f, std::min (res.z, 1.f)) * 255.f);
    
    return cv::Scalar(b, g, r);
}

// Draw motion vectors with stride given
void GestureRecognizer::drawMotionField(const cv::Mat& flow, cv::Mat& out, int stride)
{
    out = cv::Mat::zeros(m_flow.rows, m_flow.cols, CV_8UC3);
    
    for (size_t y = 0; y < flow.rows; y += stride)
    {
        for (size_t x = 0; x < flow.cols; x += stride)
        {
            cv::Point2f vv = flow.at<cv::Point2f>(y, x);
            cv::Point2i p1(x, y);
            cv::Point2i p2(int(x + vv.x), int(y + vv.y));
            cv::arrowedLine(out, p1, p2, jet(vv), 1);
        }
    }
}

void GestureRecognizer::drawHeatmap(const cv::Mat& flow, cv::Mat& out)
{
    cv::Mat flow_planes[2];
    cv::split(flow, flow_planes);
    cv::Mat magnitude, angle;
    cv::cartToPolar(flow_planes[0], flow_planes[1], magnitude, angle, true);
    cv::threshold(magnitude, magnitude, 1, 999, cv::THRESH_TOZERO);
    //cv::normalize(magnitude, out, 0.0f, 1.0f, cv::NORM_MINMAX);

    magnitude.convertTo(magnitude, CV_8UC1);
    cv::applyColorMap(magnitude, out, cv::COLORMAP_JET);
}