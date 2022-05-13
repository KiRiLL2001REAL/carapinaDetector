#include <iostream>
#include <opencv2/opencv.hpp>
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>

#include "extra.hpp"

struct TLine {
    double angle;
    cv::Point2d vec;
    cv::Point2d pos[2];
    cv::Point2d straight[2];

    //undefined line
    TLine() {}

    TLine(cv::Point2d p0, cv::Point2d p1, int maxX, int maxY) {
        pos[0] = p0;
        pos[1] = p1;
        
        vec = p1 - p0;
        double lenNormCoeff = 1.0 / sqrt(vec.x * vec.x + vec.y * vec.y);
        vec.x *= lenNormCoeff;
        vec.y *= lenNormCoeff;
                
        angle = atan2(vec.y, vec.x);

        double sy0 = ((0.0 - p0.x) * vec.y) / vec.x + p0.y;           // Y при X = 0
        double sy1 = ((double(maxX) - p0.x) * vec.y) / vec.x + p0.y;  // Y при X = maxX
        double sx0 = ((0.0 - p0.y) * vec.x) / vec.y + p0.x;           // X при Y = 0
        double sx1 = ((double(maxY) - p0.y) * vec.x) / vec.y + p0.x;  // X при Y = maxY

        std::vector<double> v;
        // при X = 0, Y не выходит за пределы изображения
        if (sy0 <= maxY && sy0 >= 0) { v.push_back(0);    v.push_back(sy0); }
        // при X = maxX, Y не выходит за пределы изображения
        if (sy1 <= maxY && sy1 >= 0) { v.push_back(maxX); v.push_back(sy1); }
        // при Y = 0, X не выходит за пределы изображения
        if (sx0 <= maxX && sx0 >= 0) { v.push_back(sx0); v.push_back(0);    }
        // при Y = maxY, X не выходит за пределы изображения
        if (sx1 <= maxX && sx1 >= 0) { v.push_back(sx1); v.push_back(maxY); }

        straight[0].x = v[0];
        straight[0].y = v[1];
        straight[1].x = v[2];
        straight[1].y = v[3];
    }

    double dist(const TLine& right) {
        using namespace cv;
        /*
        Point2d c0 = straight[1] + straight[0];
        c0 /= 2;
        c0.x = abs(c0.x);
        c0.y = abs(c0.y);
        Point2d c1 = right.straight[1] + right.straight[0];
        c1 /= 2;
        c1.x = abs(c1.x);
        c1.y = abs(c1.y);
        */

        //Point2d perpV = vec;
        //std::swap(perpV.x, perpV.y);
        //perpV.y *= -1;

        Point2d c = straight[0] + straight[1];
        c /= 2;

        double x1;
        double y1;

        if (vec.x >= 0.5) {
            y1 = c.y + 1;
            x1 = ((0.0 - c.y) * vec.x) / vec.y + c.x;
        } else {
            x1 = c.x + 1;
            y1 = ((0.0 - c.x) * vec.y) / vec.x + c.y;
        }

        TLine perpL = TLine(c, Point2d(x1, y1), 99999, 99999);

        cv::Point2d cpoint;
        extra::cross(cpoint, right.straight[0], right.straight[1], perpL.straight[0], perpL.straight[1]);

        double dx = c.x - cpoint.x;
        double dy = c.y - cpoint.y;

        return sqrt(dx * dx + dy * dy);
    }
};

const std::string DATASET_PATH = "C:\\Dataset";

void handleMatrix(const cv::Mat& src, cv::Mat& dst, cv::Mat& rescaled);
cv::Mat laplasiaze(const cv::Mat& src);

int main()
{
    using namespace cv;
    TLine l1 = TLine(Point2d(0, 2), Point2d(2, 0), 6, 4);
    TLine l2 = TLine(Point2d(2, 4), Point2d(6, 0), 6, 4);
    std::cout << l1.dist(l2);


    return 0;

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
                        counter = imagePath.size() - 1;
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

    TLine* myLines = new TLine[linesP.size()];
    for (size_t i = 0; i < linesP.size(); i++)
    {
        Vec4i l = linesP[i];
        line(rgb, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        myLines[i] = TLine( Point2d(l[0], l[1]), Point2d(l[2], l[3]), cannyed.cols, cannyed.rows );
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