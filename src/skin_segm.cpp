#pragma once

#include "skin_segm.h"

SkinSegmentator::SkinSegmentator(float _skin_prior)
{
    if (_skin_prior <= 0)
    {
        _skin_prior = 0.01f;
        std::cout << "warning: skin prior less or equal zero: set as 0.01" << std::endl;
    }
    if (_skin_prior >= 1)
    {
        _skin_prior = 0.99f;
        std::cout << "warning: skin prior more or equal one: set as 0.99" << std::endl;
    }
    this->skin_prior = _skin_prior;
    this->nonskin_prior = 1 - _skin_prior;
}

SkinSegmentator::~SkinSegmentator() { }

float SkinSegmentator::multigauss(const float x[3], const float mu[3], const float sigma[3], float w)
{
    float det = sigma[0] * sigma[1] * sigma[2];
    if (det == 0)
    {
        return 0.0f;
    }

    float e_coeff = 0;
    float mu_dev[3] = {x[0] - mu[0], x[1] - mu[1], x[2] - mu[2]};
    float tmp[3] = {mu_dev[0] * (1/sigma[0]), mu_dev[1] * (1/sigma[1]), mu_dev[2] * (1/sigma[2])};
    e_coeff = tmp[0] * mu_dev[0] + tmp[1] * mu_dev[1] + tmp[2] * mu_dev[2];
    e_coeff *= -0.5;

    float e = expf(e_coeff);

    float gauss = w * (e / sqrtf(powf(TWOPI, 3) * det));

    return gauss;
}


float SkinSegmentator::skin_likelihood(const float pixel[3])
{
    float lhood = 0;
    for (int mode = 0; mode < 16; mode++)
    {
        const float _mean[3] = {Skin_Mus[mode][0], Skin_Mus[mode][1], Skin_Mus[mode][2]};
        const float _sigma[3] = {Skin_Sigmas[mode][0], Skin_Sigmas[mode][1], Skin_Sigmas[mode][2]};
        lhood += multigauss(pixel, _mean, _sigma, Skin_Ws[mode]);
    }
    return lhood;
}


float SkinSegmentator::nonskin_likelihood(const float pixel[3])
{
    float lhood = 0;
    for (int mode = 0; mode < 16; mode++)
    {
        const float _mean[3] = {Nonskin_Mus[mode][0], Nonskin_Mus[mode][1], Nonskin_Mus[mode][2]};
        const float _sigma[3] = {Nonskin_Sigmas[mode][0], Nonskin_Sigmas[mode][1], Nonskin_Sigmas[mode][2]};
        lhood += multigauss(pixel, _mean, _sigma, Nonskin_Ws[mode]);
    }
    return lhood;
}


float SkinSegmentator::segment_skin_pixel(const float pixel[3])
{
    float skin_prob = skin_likelihood(pixel) * skin_prior;
    float nonskin_prob = nonskin_likelihood(pixel) * nonskin_prior;
    float denom = skin_prob + nonskin_prob;
    if (denom == 0)
    {
        return 0.0f;
    }
    else
    {
        return (skin_prob / (denom));
    }
}


bool SkinSegmentator::segment_skin(const cv::Mat& _img, cv::Mat& out, const cv::Mat& _mask)
{
    CV_Assert(!_img.empty());

    cv::Mat mask;
    if (_mask.empty())
    {
        mask = cv::Mat::ones(_img.rows, _img.cols, CV_8UC1);
    }
    else
    {
        CV_Assert(_mask.type() == CV_8UC1);

        mask = _mask;
    }
    

    cv::Mat img;
    cv::cvtColor(_img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3);

    const int w = img.cols;
    const int h = img.rows;
    out = cv::Mat::zeros(h, w, CV_32FC1);
    cv::parallel_for_(cv::Range(0, h * w), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; i++)
        {
            /* filter by mask */
            int mask_pixel = (int)mask.at<unsigned char>(i / w, i % w);
            if (mask_pixel == 0) continue;

            cv::Vec3f pixel = img.at<cv::Vec3f>(i / w, i % w);
            const float fpixel[3] = {pixel[0], pixel[1], pixel[2]};
            out.at<float>(i / w, i % w) = segment_skin_pixel(fpixel);
        }
    }, OPENCV_THREADS);

    return true;
}