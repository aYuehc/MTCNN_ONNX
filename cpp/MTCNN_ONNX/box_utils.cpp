#include "box_utils.h"


void preprocess(const Mat &img, float *out_array, int batch_num) {
	//cvtColor(img, img, cv::COLOR_BGR2RGB);
	int batch_offset;
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				batch_offset = batch_num * img.channels() * img.rows * img.cols;
				out_array[batch_offset + (c * img.rows * img.cols + i * img.cols + j)] = (img.ptr<uchar>(i)[j * 3 + c] - 127.5f) * 0.0078125f;
				//out_array[batch_offset + (c * img.rows * img.cols + i * img.cols + j)] = img.ptr<uchar>(i)[j * 3 + c];
			}
		}
	}
};

void nms(vector<vector<float>> &srcbboxes, vector<vector<float>> &resbboxes,
	float overlap_threshold, string mode) {
	if (srcbboxes.size() == 0) return;

	const size_t size = srcbboxes.size();
	if (!size) return;

	multimap<float, size_t> idxs;
	for (size_t i = 0; i < size; ++i) {
		idxs.insert(pair<float, size_t>(srcbboxes[i][4], i));
	}

	while (idxs.size() > 0) {
		auto lastElem = --end(idxs);
		const vector<float>& bbox = srcbboxes[lastElem->second];
		const Rect& rect1 = Rect(
			srcbboxes[lastElem->second][0], 
			srcbboxes[lastElem->second][1], 
			srcbboxes[lastElem->second][2] - srcbboxes[lastElem->second][0] + 1.0f,
			srcbboxes[lastElem->second][3] - srcbboxes[lastElem->second][1] + 1.0f);
		resbboxes.push_back(bbox);

		idxs.erase(lastElem);

		for (auto pos = begin(idxs); pos != end(idxs); ) {
			const Rect& rect2 = Rect(
				srcbboxes[pos->second][0],
				srcbboxes[pos->second][1],
				srcbboxes[pos->second][2] - srcbboxes[pos->second][0] + 1.0f,
				srcbboxes[pos->second][3] - srcbboxes[pos->second][1] + 1.0f);

			float intArea = (rect1 & rect2).area();
			float overlap;

			// if there is sufficient overlap, suppress the current bounding box
			if (mode == "union") {
				float unionArea = rect1.area() + rect2.area() - intArea;
				overlap = intArea / unionArea;
				//if (overlap > overlap_threshold) pos = idxs.erase(pos);
				//else pos++;
			}
			else if (mode == "min") {
				overlap = intArea / min(rect1.area(), rect2.area());
			}

			if (overlap > overlap_threshold) pos = idxs.erase(pos);
			else pos++;
			//else if (mode == "min") cout << "The nms by min is being developed." << endl;
		}
	}
	srcbboxes.clear();
};

void calibration(vector<vector<float>> &src_bboxes, vector<vector<float>> &res_bboxes) {
	res_bboxes = vector<vector<float>>(src_bboxes.size());
	vector<float> sqr_bbox(4);
	int ind = 0;

	for (auto& bbox : src_bboxes) {
		float off_x1 = (bbox[2] - bbox[0] + 1.0f) * bbox[5];
		float off_y1 = (bbox[3] - bbox[1] + 1.0f) * bbox[6];
		float off_x2 = (bbox[2] - bbox[0] + 1.0f) * bbox[7];
		float off_y2 = (bbox[3] - bbox[1] + 1.0f) * bbox[8];

		bbox[0] += off_x1;
		bbox[1] += off_y1;
		bbox[2] += off_x2;
		bbox[3] += off_y2;

		res_bboxes[ind] = bbox;
		ind++;
	}
};

void calibration_and_ToSquare(vector<vector<float>> &src_bboxes, vector<vector<float>> &sqr_bboxes) {
	sqr_bboxes = vector<vector<float>>(src_bboxes.size());
	vector<float> sqr_bbox(4);
	int ind = 0;

	for (auto& bbox : src_bboxes) {
		float off_x1 = (bbox[2] - bbox[0] + 1.0f) * bbox[5];
		float off_y1 = (bbox[3] - bbox[1] + 1.0f) * bbox[6];
		float off_x2 = (bbox[2] - bbox[0] + 1.0f) * bbox[7];
		float off_y2 = (bbox[3] - bbox[1] + 1.0f) * bbox[8];

		bbox[0] += off_x1;
		bbox[1] += off_y1;
		bbox[2] += off_x2;
		bbox[3] += off_y2;

		// Convert to square
		float w = bbox[2] - bbox[0] + 1;
		float h = bbox[3] - bbox[1] + 1;

		if (w >= h) {
			sqr_bbox[0] = bbox[0];
			sqr_bbox[1] = bbox[1] + 0.5f * (h - w);
			sqr_bbox[2] = sqr_bbox[0] + w - 1.0f;
			sqr_bbox[3] = sqr_bbox[1] + w - 1.0f;
		}
		else {
			sqr_bbox[0] = bbox[0] + 0.5f * (w - h);
			sqr_bbox[1] = bbox[1];
			sqr_bbox[2] = sqr_bbox[0] + h - 1.0f;
			sqr_bbox[3] = sqr_bbox[1] + h - 1.0f;
		}

		sqr_bboxes[ind] = sqr_bbox;
		ind++;
	}
};

void get_img_boxes(vector<vector<float>> &srcbboxes, const Mat &img, float *out_array, int size) {
	int num_boxes = srcbboxes.size();

	// correct_bboxes
	Rect img_rect = Rect({}, img.size());
	Rect roi;
	Rect intersection;
	Rect inter_roi;
	Mat crop;
	int cnt = 0;
	for (auto& bbox : srcbboxes) {
		if (bbox[0] < 0 || bbox[1] < 0 || bbox[2] > img.cols || bbox[3] > img.rows) {
			roi = Rect(bbox[0], bbox[1], (bbox[2] - bbox[0]) + 1, (bbox[3] - bbox[1]) + 1);
			intersection = img_rect & roi;
			inter_roi = intersection - roi.tl();
			crop = Mat::zeros(roi.size(), img.type());
			img(intersection).copyTo(crop(inter_roi));
		}
		else {
			roi = Rect(bbox[0], bbox[1], (bbox[2] - bbox[0]) + 1, (bbox[3] - bbox[1]) + 1);
			img(roi).copyTo(crop);
		}

		resize(crop, crop, Size(size, size), INTER_LINEAR);
		preprocess(crop, out_array, cnt);
		cnt++;
	}
}