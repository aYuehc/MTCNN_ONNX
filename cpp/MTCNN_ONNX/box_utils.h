#pragma once
#ifndef BOX_UTILS_H
#define BOX_UTILS_H

#include <opencv2/core/core.hpp>  
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
#include <vector>
#include <string>
#include <stdlib.h>
#include <iostream> 


using namespace cv;
using namespace std;

void preprocess(const Mat&, float*, int);

void nms(vector<vector<float>>&, vector<vector<float>>&, float, string);

void calibration(vector<vector<float>>&, vector<vector<float>>&);

void calibration_and_ToSquare(vector<vector<float>>&, vector<vector<float>>&);

void get_img_boxes(vector<vector<float>>&, const Mat&, float*, int);

#endif // !BOX_UTILS_H
