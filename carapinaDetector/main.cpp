#include <iostream>
#include <opencv2/opencv.hpp>
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>

#include "extra.hpp"

const std::string DATASET_PATH = "D:\\Dataset";



int main()
{
    using namespace std;

    const int WIN_WIDTH = 1200;
    const int WIN_HEIGHT = 700;

    const int LOW_TRESHOLD = 20;
    const int HIGH_TRESHOLD = 70;

    vector<string> imagePath = {};

    extra::loadFilenames(DATASET_PATH, ".bmp", imagePath);
    //for (auto& it : imagePath)
    //    cout << it << '\n';

    if (imagePath.empty()) {
        cout << "Directory \"" << DATASET_PATH << "\" is empty or doesn't exist.\n";
        return 0;
    }

    cv::Mat filtered;
    cv::Mat grayMat = cv::imread(imagePath[0], cv::IMREAD_GRAYSCALE);
    //cv::blur(grayMat, grayMat, cv::Size(5, 5));
    cv::blur(grayMat, filtered, cv::Size(7, 7));
    //cv::GaussianBlur(grayMat, grayMat, cv::Size(3, 3), 5);

    //cv::bilateralFilter(grayMat, filtered, 5, 11, 17);
    cv::Canny(filtered, filtered, LOW_TRESHOLD, HIGH_TRESHOLD, 3);
    cv::imwrite(imagePath[0] + ".jpg", filtered);

    sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "Carapina detector", sf::Style::Close);
    sf::Clock timer;
    sf::Image image1;
    sf::Image image2;
    sf::Texture texture1;
    sf::Texture texture2;
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

    cv::Mat tmpMat;
    cv::cvtColor(grayMat, tmpMat, cv::COLOR_GRAY2RGBA);
    image1.create(tmpMat.cols, tmpMat.rows, tmpMat.ptr());
    tmpMat.release();
    cv::cvtColor(filtered, tmpMat, cv::COLOR_GRAY2RGBA);
    image2.create(tmpMat.cols, tmpMat.rows, tmpMat.ptr());
    tmpMat.release();
    texture1.loadFromImage(image1);
    texture2.loadFromImage(image2);
    sprite.setTexture(texture1);

    //float scale = float(WIN_HEIGHT) / sprite.getLocalBounds().height;
    //sprite.setScale({ scale, scale });

    float scale = (float(WIN_WIDTH) / sprite.getLocalBounds().width) / 2;
    sprite.setScale({ scale, scale });

    int counter = 1;
    const int imgDelayMillis = 0;

    timer.restart();
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
                if (e.key.code == sf::Keyboard::Escape)
                    window.close();
                break;
            }
        }

        if (timer.getElapsedTime().asMilliseconds() >= imgDelayMillis) {
            if (counter < imagePath.size()) {
                grayMat.release();
                grayMat = cv::imread(imagePath[counter], cv::IMREAD_GRAYSCALE);
                filtered.release();
                //cv::blur(grayMat, grayMat, cv::Size(5, 5));
                cv::blur(grayMat, filtered, cv::Size(7, 7));
                //cv::GaussianBlur(grayMat, grayMat, cv::Size(3, 3), 5);
                //cv::bilateralFilter(grayMat, filtered, 5, 11, 17);
                cv::Canny(filtered, filtered, LOW_TRESHOLD, HIGH_TRESHOLD, 3);
                cv::imwrite(imagePath[counter] + ".jpg", filtered);

                cv::RNG rng(12345);
                vector<vector<cv::Point> > contours;
                vector<cv::Vec4i> hierarchy;
                findContours(filtered, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_KCOS);
                cv::Mat drawing = cv::Mat::zeros(filtered.size(), CV_8UC3);
                for (size_t i = 0; i < contours.size(); i++)
                {
                    cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
                    drawContours(drawing, contours, (int)i, color, 2, cv::LINE_8, hierarchy, 0);
                }
                cv::imwrite(imagePath[counter] + "1.jpg", drawing);
                drawing.release();

                //vector<cv::Vec2f> lines; // will hold the results of the detection
                //cv::HoughLines(filtered, lines, 0.5, CV_PI / 360, 150, 0, 0); // runs the actual detection
                //// Draw the lines
                //for (size_t i = 0; i < lines.size(); i++)
                //{
                //    float rho = lines[i][0], theta = lines[i][1];
                //    cv::Point pt1, pt2;
                //    double a = cos(theta), b = sin(theta);
                //    double x0 = a * rho, y0 = b * rho;
                //    pt1.x = cvRound(x0 + filtered.cols * (-b));
                //    pt1.y = cvRound(y0 + filtered.rows * (a));
                //    pt2.x = cvRound(x0 - filtered.cols * (-b));
                //    pt2.y = cvRound(y0 - filtered.rows * (a));
                //    line(grayMat, pt1, pt2, cv::Scalar(255), 1, cv::LINE_AA);
                //}

                // Probabilistic Line Transform
                //grayMat.release();
                //grayMat = cv::Mat(filtered.rows, filtered.cols, CV_8UC1);
                //cv::Mat mat;
                //cv::cvtColor(grayMat, mat, cv::COLOR_GRAY2RGB);
                //vector<cv::Vec4i> linesP; // will hold the results of the detection
                //cv::HoughLinesP(filtered, linesP, 2, CV_PI / 90, 30, 50, 10); // runs the actual detection
                //// Draw the lines
                //for (size_t i = 0; i < linesP.size(); i++)
                //{
                //    cv::Vec4i l = linesP[i];
                //    cv::line(mat, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
                //}

                //cv::imwrite(imagePath[counter] + "1.jpg", mat);
                //mat.release();
                //cv::imwrite(imagePath[counter] + "1.jpg", grayMat);

                cv::cvtColor(grayMat, tmpMat, cv::COLOR_GRAY2RGBA);
                image1.create(tmpMat.cols, tmpMat.rows, tmpMat.ptr());
                tmpMat.release();
                cv::cvtColor(filtered, tmpMat, cv::COLOR_GRAY2RGBA);
                image2.create(tmpMat.cols, tmpMat.rows, tmpMat.ptr());
                tmpMat.release();
                texture1.loadFromImage(image1);
                texture2.loadFromImage(image2);

                text.setString(imagePath[counter]);
                cout << "Showed file \"" << imagePath[counter] << "\"\n";

                counter++;
            }
            timer.restart();
        }

        window.clear();

        sprite.setTexture(texture1);
        sprite.setPosition({ 0.f, 0.f });
        window.draw(sprite);
        sprite.setTexture(texture2);
        sprite.setPosition({ float(WIN_WIDTH) / 2, 0.f });
        window.draw(sprite);

        window.draw(text);
        window.display();

        this_thread::sleep_for(chrono::milliseconds(1));
    }

    return 0;
}