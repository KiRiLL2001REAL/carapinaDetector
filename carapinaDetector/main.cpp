#include <iostream>
#include <opencv2/opencv.hpp>
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>
#include <fstream>

#include "extra.hpp"

const std::string DATASET_PATH = "D:\\Dataset\\hackaton1";

void processGrayMat(cv::Mat& mat) {
    using namespace cv;
    using namespace std;
    sf::Clock timer;
    timer.restart();

    // === размываем изображение
    Mat blured;
    int gSize = 15;
    GaussianBlur(mat, blured, { gSize, gSize }, double(gSize) / 2);
    Mat gradX, gradY, absGradX, absGradY, grad;

    // === проходим фильтром собеля по X и по Y; полученные матрицы совмещаем
    int sobelKSize = 3;
    double scale = 1.0;
    double delta = 0.0;
    int borderType = BORDER_DEFAULT;
    // выходная матрица имеет размерность 16
    Sobel(blured, gradX, CV_16S, 1, 0, sobelKSize, scale, delta, borderType);
    Sobel(blured, gradY, CV_16S, 0, 1, sobelKSize, scale, delta, borderType);
    // приводим их к размерности 8
    convertScaleAbs(gradX, absGradX);
    convertScaleAbs(gradY, absGradY);
    addWeighted(absGradX, 0.5, absGradY, 0.5, 0, grad);

    const int LOW_TRESHOLD = 20;
    const int HIGH_TRESHOLD = 70;

    Canny(grad, grad, LOW_TRESHOLD, HIGH_TRESHOLD, 3);

    RNG rng(12345);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(grad, contours, hierarchy, RETR_TREE, CHAIN_APPROX_TC89_KCOS);
    cv::Mat drawing = Mat::zeros(grad.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++)
    {
        //Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        Scalar color = Scalar(0, 0, 255);
        drawContours(drawing, contours, (int)i, color, 1, LINE_8, hierarchy, 0);
        extra::getOrientation(contours[i], drawing);
    }

    

    gradX.release();
    gradY.release();
    absGradX.release();
    absGradY.release();
    blured.release();
    mat.release();

    grad.release();
    mat = drawing;

    cout << "Processed in " << timer.getElapsedTime().asMilliseconds() << "ms\n";
}

int main()
{
    using namespace std;

    const int WIN_WIDTH = 1200;
    const int WIN_HEIGHT = 700;

    const int LOW_TRESHOLD = 20;
    const int HIGH_TRESHOLD = 70;

    vector<string> imagePath = {};

    extra::loadFilenames(DATASET_PATH, ".bmp", imagePath);

    if (imagePath.empty()) {
        cout << "Directory \"" << DATASET_PATH << "\" is empty or doesn't exist.\n";
        return 0;
    }

    sf::ContextSettings settings;
    settings.antialiasingLevel = 16;
    sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "Carapina detector", sf::Style::Close, settings);
    sf::Clock timer;
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

    cv::Mat grayMat = cv::imread(imagePath[0], cv::IMREAD_GRAYSCALE);
    cout << "Showed file \"" << imagePath[0] << "\"\n";
    processGrayMat(grayMat);
    cv::imwrite(imagePath[0] + ".jpg", grayMat);

    extra::cvtRGBMatToImage(grayMat, image);
    texture.loadFromImage(image);
    sprite.setTexture(texture);

    float scale = float(WIN_HEIGHT) / sprite.getLocalBounds().height;
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
                cout << "Showed file \"" << imagePath[counter] << "\"\n";
                processGrayMat(grayMat);
                cv::imwrite(imagePath[counter] + ".jpg", grayMat);

                extra::cvtRGBMatToImage(grayMat, image);
                texture.loadFromImage(image);

                text.setString(imagePath[counter]);

                counter++;
            }
            timer.restart();
        }

        window.clear();

        sprite.setTexture(texture);
        window.draw(sprite);

        window.draw(text);
        window.display();

        this_thread::sleep_for(chrono::milliseconds(1));
    }

    return 0;
}