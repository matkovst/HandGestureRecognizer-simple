#pragma once

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define DEFAULT_ALPHA 1/25.f

/* The class creates exponential forgetting background and generates absdiff foreground mask */
class SimpleBackgroundSubtractor
{
public:
    SimpleBackgroundSubtractor(float _alpha = DEFAULT_ALPHA);

    ~SimpleBackgroundSubtractor();

    /* Generate foreground mask */
    bool apply(const cv::Mat& u8_frame, cv::Mat& u8_mask, bool update_bg = true);

    void euclidean_dist(const cv::Mat& f32_frame, cv::Mat& f32_out);

    void mahalanobis_dist(const cv::Mat& f32_frame, cv::Mat& f32_out);

    void getBackgroundImage(cv::Mat& out);

private:
    float alpha { DEFAULT_ALPHA };
    long long frame_num { 0 };
    cv::Mat f32_background;
    cv::Mat f32_mean;
    cv::Mat f32_stdev2;
    cv::Mat f32_prev_frame2;
    cv::Mat f32_prev_mean2;

    /* Update background */
    bool update(const cv::Mat& u8_frame);
};