#pragma once

#include <stdio.h>
#include <iostream>
#include <cmath>
#include "omp.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "weights.h"

class SkinSegmentator
{
public:
    SkinSegmentator(float _skin_prior = SKIN_PRIOR);
    ~SkinSegmentator();
    bool segment_skin(const cv::Mat& img, cv::Mat& out, const cv::Mat& _mask = cv::Mat());
    float segment_skin_pixel(const float pixel[3]);

private:
    float skin_prior { SKIN_PRIOR };
    float nonskin_prior { NONSKIN_PRIOR };

    float multigauss(const float x[3], const float mu[3], const float sigma[3], float w);
    float skin_likelihood(const float pixel[3]);
    float nonskin_likelihood(const float pixel[3]);
};