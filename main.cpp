#include <iostream>
#include <cmath>
#include "omp.h"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "src/skin_segm.h"
#include "src/bg_subtr.h"
#include "src/gesture_recognizer.h"

#define OPENCV_THREADS 4
#define WINDOW_NAME "Gesture recognition"

int run_stream(int argc, char** argv)
{
    std::cout << "info: Gesture recognition started on stream." << std::endl;

    if (argc < 2)
    {
        std::cerr << "error: Specify input path!" << std::endl;
        return -1;
    }
    std::string inputpath = argv[1];

    bool debug = false;
    if (argc > 2) debug = (atoi(argv[2]) == 1) ? true : false;

    cv::VideoCapture capture;
    if (inputpath == "0") capture.open(0);
    else if (inputpath == "1") capture.open(1);
    else capture.open(inputpath);
    if( !capture.isOpened() )
    {
        std::cerr << "error: Could not initialize capturing..." << std::endl;
        return 0;
    }

    cv::Mat frame;
    capture >> frame;
    if( frame.empty() )
    {
        std::cerr << "error: Could not capture frame..." << std::endl;
        return 0;
    }
    const int H = frame.rows;
    const int W = frame.cols;

    GestureRecognizer gRecognizer(1/(25.f*60*5));

    // main loop
    while (true)
    {
        capture >> frame;
        if( frame.empty() )
        {
            break;
        }
        
        clock_t begin = clock();
        /* ---------- CORE ---------- */

        GESTURE_TYPE gesture;
        cv::Mat debugFrame;
        gesture = gRecognizer.recognize(frame, debug, debugFrame);

        /* ---------- //// ---------- */
        clock_t end = clock();
        int elapsed = int(float(end - begin) / (CLOCKS_PER_SEC / 1000));
        // std::cout << "info: Stream elapsed: " << elapsed << std::endl;

        /* Visualize */
        if (debug)
        {
            cv::putText(frame, "det. time: " + std::to_string(elapsed) + " ms", cv::Point(10, frame.rows - 20), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 255, 0));
            cv::putText(frame, "Gesture: " + std::to_string(gesture), cv::Point(10, frame.rows - 40), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 255, 0));
            cv::Mat stacked(frame.rows, frame.cols * 2, CV_8UC3, cv::Scalar::all(0));
            frame.copyTo(stacked.colRange(0, frame.cols));
            debugFrame.copyTo(stacked.colRange(frame.cols, frame.cols * 2));
            cv::imshow(WINDOW_NAME, stacked);
        }
        else
        {
            cv::imshow(WINDOW_NAME, frame);
        }
        
        char c = (char)cv::waitKey(10);
#ifdef _WIN32
        if( c == 27 )
        {
            break;
        }
#else
        if( c == 27 )
        {
            break;
        }
#endif
    }
    capture.release();
    cv::destroyAllWindows();

    std::cout << "info: Gesture recognition finished on stream." << std::endl;
    return 0;
}


int main(int argc, char** argv)
{
    cv::setNumThreads(OPENCV_THREADS);

    int res_code = run_stream(argc, argv);

    return res_code;
}
