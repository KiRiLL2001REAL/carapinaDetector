#include <iostream>
#include <opencv2/opencv.hpp>
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>
#include <map>

#include "extra.hpp"

using namespace std;

const std::string DATASET_PATH = "D:\\data";

void processGrayMat(cv::Mat& mat, int c, const std::string& path/*, CNN_Controller& cnnc*/) {
    using namespace cv;
    using namespace std;
    sf::Clock timer;
    timer.restart();

    Mat blured;
    int bSize = 7;
    blur(mat, blured, Size(bSize, bSize));
    Canny(blured, blured, 20, 50, 3);

    string catalog = "D:\\data\\out\\" + to_string(c);
    filesystem::create_directories(catalog);

    Mat rgb;
    cvtColor(mat, rgb, COLOR_GRAY2RGB);
    // Probabilistic Line Transform
    vector<cv::Vec4i> linesP; // will hold the results of the detection
    cv::HoughLinesP(blured, linesP, 2, CV_PI / 90, 30, 40, 15); // runs the actual detection
    // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        cv::Vec4i l = linesP[i];
        cv::line(rgb, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        int cx = (l[0] + l[2]) / 2;
        int cy = (l[1] + l[3]) / 2;
        int x0 = cx - 16;
        int y0 = cy - 16;
        int x1 = cx + 17;
        int y1 = cy + 17;
        if (x0 < 0 || y0 < 0 || x1 >= mat.size().width || y1 >= mat.size().height)
            continue;
        Rect2i crop = Rect2i(Point2i(x0, y0), Point2i(x1, y1));
        imwrite(catalog + "\\" + to_string(i) + ".bmp", mat(crop));
    }
    imwrite("D:\\data\\out\\" + to_string(c) + ".jpg", rgb);

    cvtColor(rgb, rgb, COLOR_RGB2BGR);

    mat.release();
    blured.release();
    mat = rgb;

    cout << "Processed in " << timer.getElapsedTime().asMilliseconds() << "ms\n";
}

int main()
{
    using namespace std;

    const int WIN_WIDTH = 1200;
    const int WIN_HEIGHT = 700;

    vector<string> imagePath = {};

    extra::loadFilenames(DATASET_PATH, ".bmp", imagePath);
    //for (auto& it : imagePath)
    //    cout << it << '\n';

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
    processGrayMat(grayMat, 0, imagePath[0]/*, controller*/);
    //cv::imwrite(imagePath[0] + ".jpg", grayMat);

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
                processGrayMat(grayMat, counter, imagePath[counter]/*, controller */ );
                //cv::imwrite(imagePath[counter] + ".jpg", grayMat);

                extra::cvtRGBMatToImage(grayMat, image);
                texture.loadFromImage(image);

                text.setString(imagePath[counter]);

                counter++;
            }
            else text.setString("Finished");
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