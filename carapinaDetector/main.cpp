#include <iostream>
#include <opencv2/opencv.hpp>
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>

#include "TLine.hpp"
#include "extra.hpp"


const std::string DATASET_PATH = "C:\\Dataset";

void handleMatrix(const cv::Mat& src, cv::Mat& dst, cv::Mat& rescaled);
cv::Mat laplasiaze(const cv::Mat& src);

int main()
{
    /*
    using namespace cv;
    TLine::maxX = 6;
    TLine::maxY = 4;

    TLine l1 = TLine(Point2d(0, 0), Point2d(2, 2));
    TLine l2 = TLine(Point2d(0, 1), Point2d(2, 3));

    Point2d cpoint;
    if (extra::cross(cpoint, l1.pos[0], l1.pos[1], l2.pos[0], l2.pos[1]))
        std::cout << "intersection: " << cpoint << '\n';
    else
        std::cout << "distance: " << l1.dist(l2) << '\n';
    
    std::cin.get();
    return 0;
    */
    using namespace std;

    const int WIN_WIDTH = 1200;
    const int WIN_HEIGHT = 700;

    vector<string> imagePath = {};

    extra::loadFilenames(DATASET_PATH, ".jpg", imagePath);

    if (imagePath.empty()) {
        cout << "Directory \"" << DATASET_PATH << "\" is empty or doesn't exist.\n";
        return 0;
    }

    sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "Carapina detector", sf::Style::Close);
    //sf::Clock timer;
    sf::Image image;
    sf::Texture texture;
    sf::Sprite sprite;
    sf::Font font;
    font.loadFromFile("C:\\Windows\\Fonts\\courbd.ttf");
    sf::Text text;
    text.setString(imagePath[0]);
    text.setFont(font);
    text.setFillColor(sf::Color::White);
    text.setOutlineThickness(1.5);
    text.setOutlineColor(sf::Color::Black);
    text.setCharacterSize(28);

    cv::namedWindow("rescaledMat");

    cv::Mat grayMat = cv::imread(imagePath[0], cv::IMREAD_GRAYSCALE);
    cv::Mat smallMat;
    cv::Mat rescaled;
    //handleMatrix(grayMat, smallMat, rescaled);
    //cv::imshow("rescaledMat", rescaled);

    cv::Mat tmpMat;
    cv::cvtColor(grayMat, tmpMat, cv::COLOR_GRAY2RGBA);
    image.create(tmpMat.cols, tmpMat.rows, tmpMat.ptr());
    tmpMat.release();
    texture.loadFromImage(image);
    sprite.setTexture(texture);

    float scale = float(WIN_WIDTH) / sprite.getLocalBounds().width;
    sprite.setScale({ scale, scale });

    int counter = 0;
    const int imgDelayMillis = 0;

    bool finished = false;
    bool needUpdate = true;
    //timer.restart();
    while (window.isOpen())
    {
        sf::Event e;
        while (window.pollEvent(e))
        {
            switch (e.type) {
            case sf::Event::Closed:
                window.close();
                break;
            case sf::Event::KeyPressed:
                switch (e.key.code) {
                case sf::Keyboard::Escape:
                    window.close();
                    break;
                case sf::Keyboard::Right:
                    needUpdate = true;
                    counter++;
                    if (counter >= imagePath.size())
                        counter = 0;
                    break;
                case sf::Keyboard::Left:
                    needUpdate = true;
                    counter--;
                    if (counter < 0)
                        counter = (int)imagePath.size() - 1;
                    break;
                }
            }
        }

//        if (timer.getElapsedTime().asMilliseconds() >= imgDelayMillis) {
//            if (counter < imagePath.size()) {
        if (needUpdate) {
            needUpdate = false;

            grayMat.release();
            grayMat = cv::imread(imagePath[counter], cv::IMREAD_GRAYSCALE);
            smallMat.release();
            rescaled.release();
            handleMatrix(grayMat, smallMat, rescaled);
            cv::imshow("rescaledMat", rescaled);

            cv::cvtColor(grayMat, tmpMat, cv::COLOR_GRAY2RGBA);
            image.create(tmpMat.cols, tmpMat.rows, tmpMat.ptr());
            tmpMat.release();
            texture.loadFromImage(image);

            text.setString(imagePath[counter]);
            cout << "Showed file \"" << imagePath[counter] << "\"\n";
        }
                //counter++;
//            }
//            else {
//                if (!finished) {
//                    text.setString("finished");
//                    cout << "Finished.\n";
//                }
//                finished = true;
//            }
//
//            timer.restart();
//        }

        window.clear();

        sprite.setTexture(texture);
        window.draw(sprite);

        window.draw(text);
        window.display();

        this_thread::sleep_for(chrono::milliseconds(1));
    }

    cv::destroyAllWindows();

    return 0;
}



void handleMatrix(const cv::Mat& src, cv::Mat& dst, cv::Mat& rescaled) {
    using namespace cv;
    using namespace std;
    
    if (!dst.empty())
        dst.release();
    if (!rescaled.empty())
        rescaled.release();

    Mat smallMat;
    //resize(laplasiaze(src), smallMat, Size(), 0.04, 0.04, INTER_LINEAR);
    GaussianBlur(src, smallMat, Size(51, 51), 0, 0, BORDER_DEFAULT);
    namedWindow("gauss blur 15", WINDOW_NORMAL);
    imshow("gauss blur 15", smallMat);
    //resize(src, smallMat, Size(), 0.04, 0.04, INTER_LINEAR);
    resize(smallMat, smallMat, Size(), 0.04, 0.04, INTER_LINEAR);
    
    Mat cannyed;
    int cApertureSize = 3;
    double cThres1 = 20;
    double cThres2 = 40;
    Canny(smallMat, cannyed, cThres1, cThres2, cApertureSize);

    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    dilate(cannyed, cannyed, morphKernel);
    erode(cannyed, cannyed, morphKernel);

    Mat rgb;
    cvtColor(cannyed, rgb, COLOR_GRAY2RGB);

    vector<Vec4i> linesP; // will hold the results of the detection
    double rho = 0.5;
    double theta = CV_PI / (180 / rho);
    HoughLinesP(cannyed, linesP, rho, theta, 20, 30, 5); // runs the actual detection

    TLine::maxX = cannyed.cols;
    TLine::maxY = cannyed.rows;
    TLine* myLines = new TLine[linesP.size()];
    for (size_t i = 0; i < linesP.size(); i++)
    {
        Vec4i l = linesP[i];
        line(rgb, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        myLines[i] = TLine( Point2d(l[0], l[1]), Point2d(l[2], l[3]) );
    }

    qsort(
        myLines,
        linesP.size(),
        sizeof(TLine),
        [](const void* a, const void* b) {
            const TLine* l1 = static_cast<const TLine*>(a);
            const TLine* l2 = static_cast<const TLine*>(b);
            if (l1->angle < l2->angle) return -1;
            if (l1->angle > l2->angle) return 1;
            return 0;
        });



    resize(rgb, rescaled, Size(512, 512), 0, 0, INTER_LINEAR);
    cannyed.release();
    rgb.release();
    dst = smallMat;
}

cv::Mat laplasiaze(const cv::Mat& src) {
    using namespace cv;

    Mat kernel = (Mat_<float>(3, 3) <<
        1, 1, 1,
        1, -8, 1,
        1, 1, 1);

    //Mat imgLaplacian;
    //filter2D(src, imgLaplacian, CV_32F, kernel);
    //Mat sharp;
    //src.convertTo(sharp, CV_32F);
    //Mat imgResult = sharp - imgLaplacian;
    //// convert back to 8bits gray scale 
    //imgResult.convertTo(imgResult, CV_8UC3);
    //imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    //// imshow( "Laplace Filtered Image", imgLaplacian ); 
    //namedWindow("New Sharped Image", WINDOW_NORMAL); // Create a window to display results 
    //imshow("New Sharped Image", imgResult);

    // Create binary image from source image 
    Mat bw = src.clone();
    //Mat bw = imgResult;
    threshold(bw, bw, 40, 255, THRESH_BINARY);// | THRESH_OTSU);
    //adaptiveThreshold(bw, bw, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 5, 0); 

    medianBlur(bw, bw, 15);

    namedWindow("Binary Image", WINDOW_NORMAL); // Create a window to display results 
    imshow("Binary Image", bw);

    return bw;
}