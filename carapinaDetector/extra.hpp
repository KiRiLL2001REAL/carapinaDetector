#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/core.hpp>

namespace extra {
	using namespace std;

	// Получение пути файлов с заданным расширением в вектор из указанного каталога, включая подкаталоги
	void loadFilenames(const string& folder, const string& extension, vector<string>& out);
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
    float n;
    if (p2.y - p1.y != 0) {  // a(y)
        float q = (p2.x - p1.x) / (p1.y - p2.y);
        float sn = (p3.x - p4.x) + (p3.y - p4.y) * q;
        if (!sn) // c(x) + c(y)*q
            return false;
        float fn = (p3.x - p1.x) + (p3.y - p1.y) * q;   // b(x) + b(y)*q
        n = fn / sn;
    }
    else {
        if (!(p3.y - p4.y)) // b(y)
            return false;
        n = (p3.y - p1.y) / (p3.y - p4.y);   // c(y)/b(y)
    }
    dot.x = p3.x + (p4.x - p3.x) * n;  // x3 + (-b(x))*n
    dot.y = p3.y + (p4.y - p3.y) * n;  // y3 +(-b(y))*n
    return true;
}