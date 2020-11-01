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

int run_stream(int argc, char** argv)
{
    std::cout << "info: Gesture recognition started on stream." << std::endl;

    if (argc < 2)
    {
        std::cerr << "error: Specify input path!" << std::endl;
        return -1;
    }
    std::string inputpath = argv[1];

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

    GestureRecognizer gRecognizer(1/(25.f*60));
    bool visualize_debug = true;

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

        gRecognizer.recognize(frame, visualize_debug);

        /* ---------- //// ---------- */
        clock_t end = clock();
        double elapsed = double(end - begin) / CLOCKS_PER_SEC;
        // std::cout << "info: Stream elapsed: " << elapsed << std::endl;

        if (!visualize_debug) cv::imshow("Stream", frame);
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
