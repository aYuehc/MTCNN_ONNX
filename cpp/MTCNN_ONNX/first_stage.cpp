#include "first_stage.h"
#include "utils.h"

void run_first_stage(const Mat& img, Ort::Value &input_tensor, float& scale) {
	Mat simg;
	resize(img, simg, Size(ceil(img.cols * scale), ceil(img.rows * scale)));
	float* input_image;
	array<int64_t, 4> input_shape;

	input_shape = { 1, 3, simg.rows, simg.cols };
	input_image = new float[simg.channels() * simg.total()];

	preprocess(simg, input_image);

}