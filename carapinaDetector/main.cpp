#include <iostream>

#include <opencv2/opencv.hpp>
#include <SFML/Graphics.hpp>

int main()
{
    using namespace std;

    cout << "Hello OpenCV!\n" <<
        "Press any key on the graphic window close it.\n";
    
    // OpenCV lib test
    auto mat = cv::Mat(300, 300, CV_8UC3);
    for (int i = 0; i < 300; i++)
        for (int j = 0; j < 300; j++)
            if (i <= 10 || i >= 290 || j <= 10 || j >= 290) {
                static const int channels = 3;
                // помним про BGR формат пикселей
                int color[] = { 0, 0, 255 };
                for (int k = 0; k < channels; k++)
                    mat.at<cv::Vec3b>(i, j)[k] = color[k];
            }
    rectangle(mat, cv::Rect(70, 70, 160, 160), cv::Scalar(0, 128, 128), -1);

    while (cv::waitKey(1) == -1) {
        imshow("Hello from OpenCV!", mat);
    }
    cv::destroyAllWindows();



    // SFML lib test
    cout << "Hello SFML!\n" <<
        "Press any key on the graphic window close it.\n";
    sf::RenderWindow window(sf::VideoMode(300, 300), "Hello from SFML!", sf::Style::Close);
    sf::CircleShape shape(100.f);
    shape.setFillColor(sf::Color::Green);

    sf::Font font;
    font.loadFromFile("C:\\Windows\\Fonts\\courbd.ttf");
    sf::Text text;
    text.setString(L"Нажмите любую\nкнопку чтобы\nзакрыть окно");
    text.setFont(font);
    text.setFillColor(sf::Color::White);
    text.setOutlineThickness(1.5);
    text.setOutlineColor(sf::Color::Black);
    text.setCharacterSize(28);
    text.setPosition({ float((300 - text.getLocalBounds().width) / 2), float((300 - text.getLocalBounds().height) / 2) });

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            switch (event.type) {
                case sf::Event::Closed:
                case sf::Event::KeyPressed:
                    window.close();
                    break;
            }
        }

        window.clear();
        window.draw(shape);
        window.draw(text);
        window.display();

        this_thread::sleep_for(chrono::milliseconds(1));
    }
}