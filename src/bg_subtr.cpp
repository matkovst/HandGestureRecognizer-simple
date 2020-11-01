#pragma once

#include "bg_subtr.h"


SimpleBackgroundSubtractor::SimpleBackgroundSubtractor(float _alpha)
{
    if (_alpha <= 0)
    {
        _alpha = 1/25.f;
        std::cout << "warning: alpha less or equal zero: set as 0.04" << std::endl;
    }
    if (_alpha >= 1)
    {
        _alpha = 1/25.f;
        std::cout << "warning: alpha more or equal one: set as 0.04" << std::endl;
    }
    this->alpha = _alpha;
}

SimpleBackgroundSubtractor::~SimpleBackgroundSubtractor() { }

bool SimpleBackgroundSubtractor::apply(const cv::Mat& u8_frame, cv::Mat& u8_out, bool update_bg)
{
    CV_Assert (!u8_frame.empty());
    CV_Assert (u8_frame.channels() == 3);

    cv::Mat f32_frame;
    u8_frame.convertTo(f32_frame, CV_32FC3);

    if (update_bg)
    {
        bool good = update(u8_frame);
        if (!good)
        {
            return false;
        }
    }

    cv::Mat f32_out;
    euclidean_dist(f32_frame, f32_out);
    f32_out.convertTo(u8_out, CV_8UC1);
        
    return true;
}

bool SimpleBackgroundSubtractor::update(const cv::Mat& u8_frame)
{
    CV_Assert (!u8_frame.empty());
    CV_Assert (u8_frame.channels() == 3);

    frame_num++;

    cv::Mat f32_frame;
    u8_frame.convertTo(f32_frame, CV_32FC3);

    if (f32_background.empty()) // <- first frame initializes background
    {
        f32_background = f32_frame.clone();
        f32_mean = f32_frame.clone();
        f32_stdev2 = f32_frame.clone();
        cv::pow(f32_frame, 2, f32_prev_frame2);
        cv::pow(f32_mean, 2, f32_prev_mean2);
        return true;
    }
    else
    {
        f32_background = (1 - alpha) * f32_background + alpha * f32_frame;
        // cv::addWeighted(f32_mean, ((frame_num - 1)/(double)frame_num), f32_frame, (double)(1/(double)frame_num), 0, f32_mean, CV_32FC3);
        // cv::Mat f32_frame2, f32_mean2;
        // cv::pow(f32_frame, 2, f32_frame2);
        // cv::pow(f32_mean, 2, f32_mean2);
        // f32_stdev2 = f32_stdev2 + f32_prev_mean2 - f32_mean2 + (f32_frame2 - f32_stdev2 - f32_prev_mean2)/((float)frame_num);
        // cv::swap(f32_prev_frame2, f32_frame2);
        // cv::swap(f32_prev_mean2, f32_mean2);
        // //f32_background = f32_mean;
    }
        
    return true;
}

void SimpleBackgroundSubtractor::euclidean_dist(const cv::Mat& f32_frame, cv::Mat& f32_out)
{
    cv::absdiff(f32_background, f32_frame, f32_out);
    cv::transform(f32_out, f32_out, cv::Matx13f(1, 1, 1));
    
}

void SimpleBackgroundSubtractor::mahalanobis_dist(const cv::Mat& f32_frame, cv::Mat& f32_out)
{
    cv::Mat diff2 = f32_frame - f32_background;
    cv::pow(diff2, 2, diff2);
    cv::divide(diff2, f32_stdev2, f32_out, CV_32FC3);
    cv::transform(f32_out, f32_out, cv::Matx13f(1, 1, 1));
}

void SimpleBackgroundSubtractor::getBackgroundImage(cv::Mat& out)
{
    f32_background.convertTo(out, CV_8UC3);
}