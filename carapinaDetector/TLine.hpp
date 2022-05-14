#pragma once

#include <opencv2/core.hpp>
#include <iostream>
#include "extra.hpp"

struct TLine {
private:
    // x~=y->0,   x<y->-1,   x>y->1
    inline int smoothCompare(double x, double y);
public:
    double angle;
    cv::Point2d vec;
    cv::Point2d pos[2];
    cv::Point2d straight[2];
    static int maxX;
    static int maxY;

    //undefined line
    inline TLine();
    //main constructor
    inline TLine(const cv::Point2d& p0, const cv::Point2d& p1);

    inline double dist(const TLine& right);

    TLine operator=(const TLine& right) {
        angle = right.angle;
        vec = right.vec;
        pos[0] = right.pos[0];
        pos[1] = right.pos[1];
        straight[0] = right.straight[0];
        straight[1] = right.straight[1];
        return *this;
    }
};

int TLine::maxX = -1;
int TLine::maxY = -1;

inline int TLine::smoothCompare(double x, double y) {
    if (abs(x - y) <= extra::eps * 3) return 0;
    if (x < y) return -1;
    return 1;
}

inline TLine::TLine() {
    if (TLine::maxX == -1 || TLine::maxY == -1) {
        std::cout << "TLine::maxX or TLine::maxY doesn't initialized\n";
        exit(EXIT_FAILURE);
    }
}

inline TLine::TLine(const cv::Point2d& p0, const cv::Point2d& p1) : TLine() {
    using namespace extra;

    pos[0] = p0;
    pos[1] = p1;

    vec = p1 - p0;
    if (vec.x == 0) vec.x += eps;
    if (vec.y == 0) vec.y += eps;
    double lenNormCoeff = 1.0 / sqrt(vec.x * vec.x + vec.y * vec.y);
    vec.x *= lenNormCoeff;
    vec.y *= lenNormCoeff;

    angle = atan2(vec.y, vec.x);

    cv::Point2d pp0; // пересечение с левой границей
    cv::Point2d pp1; // пересечение с правой границей
    cv::Point2d pp2; // пересечение с верхней границей
    cv::Point2d pp3; // пересечение с нижней границей
    bool pp0b = extra::cross(pp0, p0, p1, cv::Point2d(0,           0          ), cv::Point2d(0,           TLine::maxY));
    bool pp1b = extra::cross(pp1, p0, p1, cv::Point2d(TLine::maxX, 0          ), cv::Point2d(TLine::maxX, TLine::maxY));
    bool pp2b = extra::cross(pp2, p0, p1, cv::Point2d(0,           0          ), cv::Point2d(TLine::maxX, 0          ));
    bool pp3b = extra::cross(pp3, p0, p1, cv::Point2d(0,           TLine::maxY), cv::Point2d(TLine::maxX, TLine::maxY));

    std::vector<cv::Point2d> v;
    // при X = 0, Y не выходит за пределы изображения
    if (pp0b && smoothCompare(pp0.y, 0) >= 0 && smoothCompare(pp0.y, maxY) <= 0)
        v.emplace_back(pp0);
    // при X = maxX, Y не выходит за пределы изображения
    if (pp1b && smoothCompare(pp1.y, 0) >= 0 && smoothCompare(pp1.y, maxY) <= 0)
        v.emplace_back(pp1);
    // при Y = 0, X не выходит за пределы изображения
    if (pp2b && smoothCompare(pp2.x, 0) >= 0 && smoothCompare(pp2.x, maxX) <= 0)
        v.emplace_back(pp2);
    // при Y = maxY, X не выходит за пределы изображения
    if (pp3b && smoothCompare(pp3.x, 0) >= 0 && smoothCompare(pp3.x, maxX) <= 0)
        v.emplace_back(pp3);

    // исключаем ситуацию с 2мя повторяющимися точками
    std::vector<cv::Point2d> vv;
    vv.emplace_back(v[0]);

    for (size_t i = 0; i < v.size(); i++) {
        for (size_t j = i + 1; j < v.size(); j++) {
            auto& a = v[i];
            auto& b = v[j];
            auto d = a - b;
            if (smoothCompare(d.x, 0) != 0 || smoothCompare(d.y, 0) != 0)
                vv.emplace_back(b);
            j++;
        }
    }

    straight[0] = vv[0];
    straight[1] = vv[1];
}

inline double TLine::dist(const TLine& right) {
    cv::Point2d p1 = straight[0];
    cv::Point2d p2 = straight[1];
    cv::Point2d p3 = right.straight[0];
    cv::Point2d p4 = right.straight[1];

    double add = p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y;
    double sub = p2.x * p1.y + p3.x * p2.y + p4.x * p3.y + p1.x * p4.y;
    double square = abs(add - sub) * 0.5;

    cv::Point2d vect = p1 - p2;
    double osnl = sqrt(vect.x * vect.x + vect.y * vect.y);

    return square / osnl;

    /*cv::Point2d c = straight[0] + straight[1];
    c /= 2;

    double x1;
    double y1;

    if (smoothCompare(vec.x, 0) == 0) {
        x1 = c.x + 1;
        y1 = c.y;
    }
    else if (smoothCompare(vec.y, 0) == 0) {
        x1 = c.x;
        y1 = c.y + 1;
    }
    else if (vec.x >= 0.5) {
        y1 = c.y + 1;
        x1 = ((0.0 - c.y) * vec.x) / vec.y + c.x;
    }
    else {
        x1 = c.x + 1;
        y1 = ((0.0 - c.x) * vec.y) / vec.x + c.y;
    }

    TLine perpL = TLine(c, cv::Point2d(x1, y1));

    cv::Point2d cpoint;
    if (!extra::cross(cpoint, right.straight[0], right.straight[1], perpL.straight[0], perpL.straight[1]))
        return -1;    

    double dx = c.x - cpoint.x;
    double dy = c.y - cpoint.y;

    return sqrt(dx * dx + dy * dy);*/
}