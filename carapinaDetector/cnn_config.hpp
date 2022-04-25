#pragma once

#include <opencv2/Core.hpp>
#include "json.hpp"
#include "cnn.hpp"
#include "threadPool.hpp"

class CNN_Controller {
private:
	int threadAvailable;
	ThreadPool *tp;

	const int cnnLayers;

	cnn::ConvLayer           *c1;
	cnn::ReLULayer           *r1;
	cnn::MaxPoolingLayer     *p1;
	cnn::ConvLayer           *c2;
	cnn::ReLULayer           *r2;
	cnn::FullyConnectedLayer *f3;
	cnn::FullyConnectedLayer *f4;
	cnn::FullyConnectedLayer *f5;
	cnn::Tensor              **x;

	friend void run(CNN_Controller &thiz, uint8_t threadId, const cnn::Tensor& input, cnn::Tensor& result);

public:
	CNN_Controller();
	~CNN_Controller();

	static cnn::Tensor matToTensor(const cv::Mat& m);

	int getMaxThreads() const;

	void initFromJson(const nlohmann::json& js);
	std::vector<cnn::Tensor> forward(const std::vector<cnn::Tensor>& x);
};



inline CNN_Controller::CNN_Controller() :
	cnnLayers(8)
{
	using namespace cnn;

	int avail = std::thread::hardware_concurrency();
	if (avail > 128)
		avail = 128;
	threadAvailable = avail;
	tp = new ThreadPool(threadAvailable);

	c1 = new ConvLayer[threadAvailable];
	r1 = new ReLULayer[threadAvailable];
	p1 = new MaxPoolingLayer[threadAvailable];
	c2 = new ConvLayer[threadAvailable];
	r2 = new ReLULayer[threadAvailable];
	f3 = new FullyConnectedLayer[threadAvailable];
	f4 = new FullyConnectedLayer[threadAvailable];
	f5 = new FullyConnectedLayer[threadAvailable];
	x = new Tensor*[threadAvailable];
	for (int i = 0; i < threadAvailable; i++) {
		c1[i] = ConvLayer(TensorSize(64, 64, 1), 8, 7, 3, 2);
		r1[i] = ReLULayer(c1[i].getOutputSize());
		p1[i] = MaxPoolingLayer(r1[i].getOutputSize(), 2);
		c2[i] = ConvLayer(p1[i].getOutputSize(), 4, 5, 2, 2);
		r2[i] = ReLULayer(c2[i].getOutputSize());
		f3[i] = FullyConnectedLayer(r2[i].getOutputSize(), 32, "relu");
		f4[i] = FullyConnectedLayer(f3[i].getOutputSize(), 16, "relu");
		f5[i] = FullyConnectedLayer(f4[i].getOutputSize(), 1, "sigmoid");
		x[i] = new Tensor[cnnLayers];
	}
}

inline CNN_Controller::~CNN_Controller()
{
	delete[] c1;
	delete[] r1;
	delete[] p1;
	delete[] c2;
	delete[] r2;
	delete[] f3;
	delete[] f4;
	delete[] f5;
	for (int i = 0; i < threadAvailable; i++)
		delete[] x[i];
	delete[] x;
	delete tp;
}

inline cnn::Tensor CNN_Controller::matToTensor(const cv::Mat& m)
{
	int width = m.size().width;
	int height = m.size().height;
	cnn::Tensor t(width, height, 1);
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
			t(0, i, j) = (double)m.data[i * height + j] / 255.0;
	return t;
}

inline int CNN_Controller::getMaxThreads() const
{
	return threadAvailable;
}

inline void CNN_Controller::initFromJson(const nlohmann::json& js)
{
	using namespace cnn;
	for (int i = 0; i < threadAvailable; i++) {
		c1[i] = ConvLayer(js["c1"]);
		r1[i] = ReLULayer(js["r1"]);
		p1[i] = MaxPoolingLayer(js["p1"]);
		c2[i] = ConvLayer(js["c2"]);
		r2[i] = ReLULayer(js["r2"]);
		f3[i] = FullyConnectedLayer(js["f3"]);
		f4[i] = FullyConnectedLayer(js["f4"]);
		f5[i] = FullyConnectedLayer(js["f5"]);
	}
}

void run(CNN_Controller& thiz, uint8_t threadId, const cnn::Tensor& tensor, cnn::Tensor& result)
{
	cnn::Tensor t = tensor;
	thiz.x[threadId][0] = t;
	t = thiz.c1[threadId].forward(t);
	thiz.x[threadId][1] = t;
	t = thiz.r1[threadId].forward(t);
	thiz.x[threadId][2] = t;
	t = thiz.p1[threadId].forward(t);
	thiz.x[threadId][3] = t;
	t = thiz.c2[threadId].forward(t);
	thiz.x[threadId][4] = t;
	t = thiz.r2[threadId].forward(t);
	thiz.x[threadId][5] = t;
	t = thiz.f3[threadId].forward(t);
	thiz.x[threadId][6] = t;
	t = thiz.f4[threadId].forward(t);
	thiz.x[threadId][7] = t;
	t = thiz.f5[threadId].forward(t);
	result = t;
}

inline std::vector<cnn::Tensor> CNN_Controller::forward(const std::vector<cnn::Tensor>& vec)
{
	using namespace std;
	using namespace cnn;
	vector<Tensor> result(vec.size());

	size_t index = 0;
	size_t size = vec.size();

	while (index < size) {
		for (uint8_t i = 0; i < threadAvailable && index < size; i++) {
			tp->addTask(run, ref(*this), i, ref(vec[index]), ref(result[index]));
			index++;
		}
		tp->waitAll();
	}

	return result;
}

/*
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
*/