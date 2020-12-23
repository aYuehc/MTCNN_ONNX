#include <windows.h>
#include <windowsx.h>
#include <onnxruntime_cxx_api.h>
//#include <cuda_provider_factory.h>
#include <onnxruntime_c_api.h>

#include <opencv2/core/core.hpp>  
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <vector>
#include <string>
#include <stdlib.h> 
#include <iostream> 

#include "box_utils.h"

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "onnxruntime.lib")

using namespace cv;
using namespace std;


float min_face_size = 15.0f;	//if this value is too low the algorithm will use a lot of memory
float thresholds[3] = { 0.6f, 0.7f, 0.8f };	// for probabilities
float nms_thresholds[3] = { 0.7f, 0.7f, 0.7f };	//for NMS


int main() {
	Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "MTCNN" };
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	Ort::SessionOptions session_option;
	session_option.SetIntraOpNumThreads(1);
	session_option.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	Ort::Session pnet(env, L"../weights/pnet.onnx", session_option);
	Ort::Session rnet(env, L"../weights/rnet.onnx", session_option);
	Ort::Session onet(env, L"../weights/onet.onnx", session_option);

	Ort::AllocatorWithDefaultOptions allocator;
	cout << "pnetOutut.Name:\t" << pnet.GetOutputName(0, allocator) << '\t' << pnet.GetOutputName(1, allocator) << endl;
	cout << "rnetOutut.Name:\t" << rnet.GetOutputName(0, allocator) << '\t' << rnet.GetOutputName(1, allocator) << endl;
	cout << "onetOutut.Name:\t" << onet.GetOutputName(0, allocator) << '\t' << onet.GetOutputName(1, allocator)
		<< '\t' << onet.GetOutputName(2, allocator) << endl;

	string img_name = "kingjames.jpg";
	Mat img;
	float* input_image;
	array<int64_t, 4> input_shape;
	img = imread("../images/" + img_name);
	cv::cvtColor(img, img, COLOR_BGR2RGB);
	//build an image pyramid scaling array;

	float min_length = min(img.rows, img.cols);
	float min_detection_size = 12;
	float factor = 0.707f; // sqrt(0.5)
	

	//scales the image so that
	//minimum size that we can detect equals to
	//minimum face size that we want to detect
	min_length *= (min_detection_size / min_face_size);

	vector<float> scales;
	float factor_count = 0;
	while (min_length > min_detection_size) {
		scales.push_back((min_detection_size / min_face_size) * pow(factor, factor_count));
		min_length *= factor;
		factor_count += 1;
	}

	//for (int i = 0; i < scales.size(); i++) cout << scales[i] << endl;

	vector<vector<float>> output_bboxes;
	const char* pnet_input_names[] = { "input" };
	vector<const char*>pnet_output_names(2);
	pnet_output_names[0] = { "bbox" };
	pnet_output_names[1] = { "conf" };

	for (int scales_ind = 0; scales_ind < scales.size(); scales_ind++) {
		// Run first stage.
		Mat scaled_img;
		resize(img, scaled_img, Size(ceil(img.cols * scales[scales_ind]), ceil(img.rows * scales[scales_ind])), INTER_LINEAR);
		input_shape = { 1, 3, scaled_img.rows, scaled_img.cols };
		input_image = new float[scaled_img.channels() * scaled_img.rows * scaled_img.cols];
		preprocess(scaled_img, input_image, 0);

		Ort::Value pnet_input_tensor{ nullptr };
		vector<Ort::Value>pnet_output_tensor;

		pnet_input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image, 1 * scaled_img.channels() * scaled_img.rows * scaled_img.cols, input_shape.data(), input_shape.size());
		pnet_output_tensor = pnet.Run(nullptr, pnet_input_names, &pnet_input_tensor, 1, pnet_output_names.data(), 2);
		delete[] input_image;

		float* pnet_bbox = pnet_output_tensor[0].GetTensorMutableData<float>();
		float* pnet_conf = pnet_output_tensor[1].GetTensorMutableData<float>();

		vector<int64_t> pnet_conf_shape = pnet_output_tensor[1].GetTensorTypeAndShapeInfo().GetShape();
		vector<int64_t> pnet_bbox_shape = pnet_output_tensor[0].GetTensorTypeAndShapeInfo().GetShape();

		float stride = 2;
		float cell_size = 12;
		int pnet_conf_neg_len = pnet_conf_shape[2] * pnet_conf_shape[3];
		int pnet_conf_len = pnet_conf_shape[1] * pnet_conf_shape[2] * pnet_conf_shape[3];

		vector<vector<float>> bounding_boxes;
		for (int i = pnet_conf_neg_len; i < pnet_conf_len; i++) {
			if (pnet_conf[i] > thresholds[0]) {
				float x = (i - pnet_conf_neg_len) % pnet_conf_shape[3];
				float y = (i - pnet_conf_neg_len) / pnet_conf_shape[3];

				vector<float> bounding_box;
				bounding_box.push_back((stride * x + 1.0f) / scales[scales_ind]);
				bounding_box.push_back((stride * y + 1.0f) / scales[scales_ind]);
				bounding_box.push_back((stride * x + 1.0f + cell_size) / scales[scales_ind]);
				bounding_box.push_back((stride * y + 1.0f + cell_size) / scales[scales_ind]);
				bounding_box.push_back(pnet_conf[i]);
				bounding_box.push_back(pnet_bbox[i - pnet_conf_neg_len]);
				bounding_box.push_back(pnet_bbox[i]);
				bounding_box.push_back(pnet_bbox[i + pnet_conf_neg_len * 1]);
				bounding_box.push_back(pnet_bbox[i + pnet_conf_neg_len * 2]);
				bounding_boxes.push_back(bounding_box);
			}
		}

		nms(bounding_boxes, output_bboxes, 0.5, "union");
	}
	
	vector<vector<float>> bboxes;
	nms(output_bboxes, bboxes, nms_thresholds[0], "union");
	//cout << "Number of candidates after nms: " << bboxes.size() << endl;

	//// draw bboxes
	//for (int i = 0; i < bboxes.size(); i++) {
	//	rectangle(img, Point2f(bboxes[i][0], bboxes[i][1]),
	//		Point2f(bboxes[i][2], bboxes[i][3]), Scalar(0, 255, 0), 2, 8, 0);
	//}

	// Calibration + Convert to square
	vector<vector<float>> sqr_bboxes;
	calibration_and_ToSquare(bboxes, sqr_bboxes);

	//// draw bboxes
	//for (int i = 0; i < sqr_bboxes.size(); i++) {
	//	rectangle(img, Point2f(sqr_bboxes[i][0], sqr_bboxes[i][1]),
	//		Point2f(sqr_bboxes[i][2], sqr_bboxes[i][3]), Scalar(0, 0, 255), 2, 8, 0);
	//}

	// get_image_boxes(bounding_boxes, img, size=24):
	const char* rnet_input_names[] = { "input" };
	vector<const char*>rnet_output_names(2);
	rnet_output_names[0] = { "bbox" };
	rnet_output_names[1] = { "conf" };

	int size = 24;		
	input_shape = { int(sqr_bboxes.size()), 3, size, size };
	input_image = new float[sqr_bboxes.size() * 3 * size * size];
	get_img_boxes(sqr_bboxes, img, input_image, size);
	Ort::Value rnet_input_tensor{ nullptr };
	vector<Ort::Value>rnet_output_tensor;

	rnet_input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image, sqr_bboxes.size() * 3 * size * size, input_shape.data(), input_shape.size());
	rnet_output_tensor = rnet.Run(nullptr, rnet_input_names, &rnet_input_tensor, 1, rnet_output_names.data(), 2);
	delete[] input_image;

	float* rnet_bbox = rnet_output_tensor[0].GetTensorMutableData<float>();
	float* rnet_conf = rnet_output_tensor[1].GetTensorMutableData<float>();

	vector<int64_t> rnet_conf_shape = rnet_output_tensor[1].GetTensorTypeAndShapeInfo().GetShape();
	vector<int64_t> rnet_bbox_shape = rnet_output_tensor[0].GetTensorTypeAndShapeInfo().GetShape();

	int rnet_conf_neg_len = rnet_conf_shape[0];
	int rnet_conf_len = rnet_conf_shape[0] * rnet_conf_shape[1];

	output_bboxes.clear();
	for (int i = 0; i < rnet_conf_len; i++) {
		if (i % 2) {
			if (rnet_conf[i] > thresholds[1]) {
				//cout << i << ends << i / 2 << ends << rnet_conf[i] << endl;
				vector<float> bbox;
				bbox.push_back(sqr_bboxes[i / 2][0]);
				bbox.push_back(sqr_bboxes[i / 2][1]);
				bbox.push_back(sqr_bboxes[i / 2][2]);
				bbox.push_back(sqr_bboxes[i / 2][3]);
				bbox.push_back(rnet_conf[i]);
				bbox.push_back(rnet_bbox[(i - 1) * 2]);
				bbox.push_back(rnet_bbox[(i - 1) * 2 + 1]);
				bbox.push_back(rnet_bbox[(i - 1) * 2 + 2]);
				bbox.push_back(rnet_bbox[(i - 1) * 2 + 3]);
				output_bboxes.push_back(bbox);
			}
		}
	}

	//cout << "Number of candidates: " << output_bboxes.size() << endl;

	bboxes.clear();
	nms(output_bboxes, bboxes, nms_thresholds[1], "union");
	//cout << "Number of candidates after nms: " << bboxes.size() << endl;

	sqr_bboxes.clear();
	calibration_and_ToSquare(bboxes, sqr_bboxes);
	
	const char* onet_input_names[] = { "input" };
	vector<const char*>onet_output_names(3);
	onet_output_names[0] = { "bbox" };
	onet_output_names[1] = { "conf" };
	onet_output_names[2] = { "landmark" };

	size = 48;
	input_shape = { int(sqr_bboxes.size()), 3, size, size };
	input_image = new float[sqr_bboxes.size() * 3 * size * size];
	get_img_boxes(sqr_bboxes, img, input_image, size);
	Ort::Value onet_input_tensor{ nullptr };
	vector<Ort::Value>onet_output_tensor;

	onet_input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image, sqr_bboxes.size() * 3 * size * size, input_shape.data(), input_shape.size());
	onet_output_tensor = onet.Run(nullptr, onet_input_names, &onet_input_tensor, 1, onet_output_names.data(), 3);
	delete[] input_image;

	float* onet_bbox = onet_output_tensor[1].GetTensorMutableData<float>();
	float* onet_conf = onet_output_tensor[2].GetTensorMutableData<float>();
	float* onet_landmark = onet_output_tensor[0].GetTensorMutableData<float>();

	vector<int64_t> onet_bbox_shape = onet_output_tensor[1].GetTensorTypeAndShapeInfo().GetShape();
	vector<int64_t> onet_conf_shape = onet_output_tensor[2].GetTensorTypeAndShapeInfo().GetShape();
	vector<int64_t> onet_landmark_shape = onet_output_tensor[0].GetTensorTypeAndShapeInfo().GetShape();

	//for (auto conf : onet_conf_shape) cout << conf << ends;
	//cout << endl;
	//for (auto bbox : onet_bbox_shape) cout << bbox << ends;
	//cout << endl;
	//for (auto landmark : onet_landmark_shape) cout << landmark << ends;
	//cout << endl;

	int onet_conf_neg_len = onet_conf_shape[0];
	int onet_conf_len = onet_conf_shape[0] * onet_conf_shape[1];

	output_bboxes.clear();
	for (int i = 0; i < onet_conf_len; i++) {
		if (i % 2) {
			if (onet_conf[i] > thresholds[2]) {
				//cout << i << ends << i / 2 << ends << onet_conf[i] << endl;
				vector<float> bbox;
				bbox.push_back(sqr_bboxes[i / 2][0]);
				bbox.push_back(sqr_bboxes[i / 2][1]);
				bbox.push_back(sqr_bboxes[i / 2][2]);
				bbox.push_back(sqr_bboxes[i / 2][3]);
				bbox.push_back(onet_conf[i]);
				bbox.push_back(onet_bbox[(i - 1) * 2]);
				bbox.push_back(onet_bbox[(i - 1) * 2 + 1]);
				bbox.push_back(onet_bbox[(i - 1) * 2 + 2]);
				bbox.push_back(onet_bbox[(i - 1) * 2 + 3]);
				output_bboxes.push_back(bbox);
			}
		}

	}

	cout << "Number of candidates: " << output_bboxes.size() << endl;

	//// draw bboxes
	//for (int i = 0; i < output_bboxes.size(); i++) {
	//	rectangle(img, Point2f(output_bboxes[i][0], output_bboxes[i][1]),
	//		Point2f(output_bboxes[i][2], output_bboxes[i][3]), Scalar(0, 0, 255), 2, 8, 0);
	//}

	bboxes.clear();
	calibration(output_bboxes, bboxes);

	output_bboxes.clear();
	nms(bboxes, output_bboxes, nms_thresholds[2], "min");
	cout << "Number of candidates after nms: " << output_bboxes.size() << endl;

	// draw bboxes
	for (int i = 0; i < output_bboxes.size(); i++) {
		rectangle(img, Point2f(output_bboxes[i][0], output_bboxes[i][1]),
			Point2f(output_bboxes[i][2], output_bboxes[i][3]), Scalar(0, 0, 255), 2, 8, 0);
	}

	//std::cout << "Total number of candidates after NMS: " << bboxes.size() << endl;
	cv::cvtColor(img, img, COLOR_RGB2BGR);
	cv::imwrite("../results/"+ img_name, img);

	return 0;
}