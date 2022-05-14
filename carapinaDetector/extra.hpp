#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/core.hpp>

namespace extra {
	using namespace std;

	static const double eps = 0.000001;

	// Получение пути файлов с заданным расширением в вектор из указанного каталога, включая подкаталоги
	void loadFilenames(const string& folder, const string& extension, vector<string>& out);
	// Получение точки перемечения прямых (при условии, что функция вернула true)
    bool cross(cv::Point2d& dot, cv::Point2d p1, cv::Point2d p2, cv::Point2d p3, cv::Point2d p4);
}

void extra::loadFilenames(const string& folder, const string& extension, vector<string>& out)
{
	if (!filesystem::exists(folder))
		return;
	for (auto& it : filesystem::directory_iterator(folder)) {
		if (it.is_directory()) {
			//loadFilenames(it.path().string(), extension, out);
			continue;
		}
		auto path = it.path();
		if (path.extension() == extension)
			out.push_back(path.string());
	}
}

bool extra::cross(cv::Point2d& dot, cv::Point2d p1, cv::Point2d p2, cv::Point2d p3, cv::Point2d p4) {
    if (p1.x == p2.x) p2.x += eps * 0.5;
    if (p1.y == p2.y) p2.y += eps * 0.5;
    if (p3.x == p4.x) p4.x += eps * 0.5;
    if (p3.y == p4.y) p4.y += eps * 0.5;

    double ka, kb;
    ka = (double)(p2.y - p1.y) / (double)(p2.x - p1.x); // Находим уклон LineA
    kb = (double)(p4.y - p3.y) / (double)(p4.x - p3.x); // Находим наклон LineB
    if (abs(ka - kb) <= eps)
        return false;

    dot.x = (ka * p1.x - p1.y - kb * p3.x + p3.y) / (ka - kb);
    dot.y = (ka * kb * (p1.x - p3.x) + ka * p3.y - kb * p1.y) / (ka - kb);

    return true;
}
