#pragma once

#include <opencv2/opencv.hpp>
#include "json.hpp"
#include "cnn.hpp"

using namespace std;

namespace setka {
	using namespace cnn;
	ConvLayer			c1(TensorSize(64, 64, 1), 8, 7, 3, 2);
	ReLULayer			r1(c1.getOutputSize());
	MaxPoolingLayer		p1(r1.getOutputSize(), 2);
	ConvLayer			c2(p1.getOutputSize(), 4, 5, 2, 2);
	ReLULayer			r2(c2.getOutputSize());
	FullyConnectedLayer	f3(r2.getOutputSize(), 32, "relu");
	FullyConnectedLayer	f4(f3.getOutputSize(), 16, "relu");
	FullyConnectedLayer	f5(f4.getOutputSize(), 1, "sigmoid");
	Tensor x1, x2, x3, x4, x5, x6, x7, x8;
}

nlohmann::json getJson()
{
	using namespace setka;
	nlohmann::json js = {};
	js["c1"] = c1.getJson();
	js["r1"] = r1.getJson();
	js["p1"] = p1.getJson();
	js["c2"] = c2.getJson();
	js["r2"] = r2.getJson();
	js["f3"] = f3.getJson();
	js["f4"] = f4.getJson();
	js["f5"] = f5.getJson();
	return js;
}

void loadFromJson(const nlohmann::json& js)
{
	using namespace setka;
	c1 = ConvLayer(js["c1"]);
	r1 = ReLULayer(js["r1"]);
	p1 = MaxPoolingLayer(js["p1"]);
	c2 = ConvLayer(js["c2"]);
	r2 = ReLULayer(js["r2"]);
	f3 = FullyConnectedLayer(js["f3"]);
	f4 = FullyConnectedLayer(js["f4"]);
	f5 = FullyConnectedLayer(js["f5"]);
}

cnn::Tensor forward(cnn::Tensor x)
{
	using namespace setka;
	x1 = x;
	x = c1.forward(x);
	x2 = x;
	x = r1.forward(x);
	x3 = x;
	x = p1.forward(x);
	x4 = x;
	x = c2.forward(x);
	x5 = x;
	x = r2.forward(x);
	x6 = x;
	x = f3.forward(x);
	x7 = x;
	x = f4.forward(x);
	x8 = x;
	x = f5.forward(x);
	return x;
}

void backward(cnn::Tensor dX, double learning_rate)
{	
	using namespace setka;
	dX = f5.backward(dX, x8);
	f5.updateWeights(learning_rate);
	dX = f4.backward(dX, x7);
	f4.updateWeights(learning_rate);
	dX = f3.backward(dX, x6);
	f3.updateWeights(learning_rate);
	dX = r2.backward(dX, x5);
	dX = c2.backward(dX, x4);
	c2.updateWeights(learning_rate);
	dX = p1.backward(dX, x3);
	dX = r1.backward(dX, x2);
	dX = c1.backward(dX, x1);
	c1.updateWeights(learning_rate);
}

cnn::Tensor matToTensor(const cv::Mat& m)
{
	int width = m.size().width;
	int height = m.size().height;
	cnn::Tensor t(width, height, 1);
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
			t(0, i, j) = m.data[i * height + j] / 255.0;
	return t;
}