#include "trie.hpp"
#include "trie_api.h"

#include "StringUtils.h"

#include <iostream>
#include <fstream>
#include <set>
#include <math.h>

using namespace std;

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


trie<std::wstring> t;
size_t beam_size = 30;


std::wstring codec = L" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_abcdefghijklmnopqrstuvwxyz{|}~£ÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž";

void load_dict(const char * fileName)
{
	t = trie<std::wstring>();

	std::ifstream ifs ( fileName, std::ifstream::in );
	if(!ifs.good()){
		std::cout << "Bad dictionary: " << fileName << std::endl;
		return;
	}

	for( std::string line; getline( ifs, line ); )
	{
		std::string word = cmp::StringUtils::trimStr(line);
		std::wstring wstr = cmp::StringUtils::utf8_to_wstring(word);
		for (size_t i=0; i < wstr.size(); i++)
			wstr[i] = towlower(wstr[i]);
		t.insert(wstr);
	}
}

class Hypot{
public:

	Hypot(){

	}

	Hypot(wchar_t last_char, float prob, wstring& text, wstring& text_low, float prob_acc, const trie<std::wstring>* node, std::vector<int>& splits,
			int start_pos, int end_pos):
		last_char(last_char), prob(prob), text(text), text_low(text_low), prob_acc(prob_acc), node(node), splits(splits), start_pos(start_pos), end_pos(end_pos){

	}

	std::wstring text;
	std::wstring text_low;
	float prob;
	float prob_acc;
	wchar_t last_char;
	const trie<std::wstring>* node;
	std::vector<int> splits;
	int start_pos;
	int end_pos;

};

int is_dict_word(const char* word)
{
	std::wstring wstr = cmp::StringUtils::utf8_to_wstring(word);
	for (size_t i=0; i < wstr.size(); i++)
		wstr[i] = towlower(wstr[i]);
	return (int) t.is_dict(wstr);
}

PyObject* decode_sofmax(PyArrayObject* input, int numOfDims,  npy_intp* input_dims)
{

	int seq_lengt = input_dims[0];
	int soft_max_cout = input_dims[1];

	std::vector<Hypot> l1;
	std::vector<Hypot> l2;

	std::vector<Hypot>* prev_hypo = &l1;
	std::vector<Hypot>* hypo_list = &l2;

	float* input_data = (float*) PyArray_DATA(input);

	std::wstring itext;
	std::vector<int> ispl;
	prev_hypo->push_back(Hypot(0, 1.0f, itext, itext, 1.0f, &t, ispl, -1, 0));

	for(int cx = 0; cx < seq_lengt; cx++){
		hypo_list->clear();
		std::map<std::wstring, int> hypo_map;
		for(int y = 0; y < soft_max_cout; y++){
			int c = y;
			float curren_prob = input_data[cx * soft_max_cout + y];
			for(size_t h = 0; h < prev_hypo->size(); h++){
				Hypot& hypo = (*prev_hypo)[h];

				float prob = hypo.prob * curren_prob;
				float probacc = hypo.prob_acc + curren_prob;
				std::wstring new_text_low =  hypo.text_low;
				std::wstring new_text =  hypo.text;

				const trie<std::wstring>* node = hypo.node;

				int start_pos = hypo.start_pos;
			  int end_pos = hypo.end_pos;
			  if(hypo.text.size() == 0)
			  	start_pos = cx;

				if( c > 3 and c != hypo.last_char){
						int ord = codec[c - 4];
						wchar_t text = (wchar_t) ord;
						wchar_t text_low = towlower(text);
						if (text_low == L' ' && hypo.node->flag){
							new_text_low += text_low;
							new_text += text;
							hypo_list->push_back(Hypot(c, prob * 0.8, new_text, new_text_low, probacc, &t, hypo.splits, start_pos, end_pos));
							hypo_list->back().splits.push_back(cx);
						}
						if( (text_low == L'.' || text_low == L'-' || text_low == L'!' || text_low == L'?') && hypo.node->flag  ){
							new_text_low += text_low;
							new_text += text;
							hypo_list->push_back(Hypot(c, prob * 0.8, new_text, new_text_low, probacc, &t, hypo.splits, start_pos, end_pos));

						}else{
							node = hypo.node->can_complete( text_low );
							if(node == NULL)
								continue;
							new_text_low += text_low;
							new_text += text;
						}
				}else{
					probacc = hypo.prob_acc;
				}
				if(c > 3)
					end_pos = cx;

				if( hypo_map.find(new_text_low) != hypo_map.end() ){
					int update_id = hypo_map.find(new_text_low)->second;
					assert(update_id < (*hypo_list).size());
					Hypot& hypo_update = (*hypo_list)[update_id];
					if(prob > hypo_update.prob){
						hypo_update.prob = prob;
						hypo_update.prob_acc = probacc;
						hypo_update.last_char = c;
						hypo_update.text = new_text;
					}
				}else{

					int id = hypo_list->size();
					hypo_list->push_back(Hypot(c, prob, new_text, new_text_low, probacc, node, hypo.splits, start_pos, end_pos));
					hypo_map[new_text_low] = id;

					if(node->flag){
						new_text += L" ";
						hypo_list->push_back(Hypot(c, prob * 0.8, new_text, new_text_low, probacc, &t, hypo.splits, start_pos, end_pos));
						hypo_list->back().splits.push_back(cx);
					}

				}
			}
		}


		sort(hypo_list->begin(), hypo_list->end(),
				[](const Hypot & a, const Hypot & b)
		{
				return a.prob > b.prob;
		});

		hypo_list->resize( std::min(beam_size, hypo_list->size()) );

		std::vector<Hypot>* tmp = hypo_list;
		hypo_list = prev_hypo;
		prev_hypo = tmp;
	}


	int max_hypo_size = 0;
	int max_splits = 0;
	for( size_t i = 0; i < prev_hypo->size(); i++ ){
		max_hypo_size = std::max( (int) (*prev_hypo)[i].text.size(), max_hypo_size );
		max_splits = std::max( (int) (*prev_hypo)[i].splits.size(), max_splits );
	}


	npy_intp size_pts[2];
	size_pts[0] = std::min(beam_size, prev_hypo->size() );
	size_pts[1] = max_hypo_size;

	PyArrayObject* out = (PyArrayObject *) PyArray_SimpleNew( 2, size_pts, NPY_INT );
	npy_intp size_pts2[2];
	size_pts2[0] = std::min(beam_size, prev_hypo->size() );
	size_pts2[1] = 3;
	PyArrayObject* out2 = (PyArrayObject *) PyArray_SimpleNew( 2, size_pts2, NPY_FLOAT );

	npy_intp size_pts3[2];
	size_pts3[0] = std::min(beam_size, prev_hypo->size() );
	size_pts3[1] = max_splits;
	PyArrayObject* out_splists = (PyArrayObject *) PyArray_SimpleNew( 2, size_pts3, NPY_INT );

	int* ptr = (int *) PyArray_GETPTR2(out, 0, 0);
	memset((void *) ptr, 0, size_pts[0] * size_pts[1] * sizeof(int));
	float* ptr2 = (float *) PyArray_GETPTR2(out2, 0, 0);
	memset((void *) ptr2, 0, size_pts2[0] * size_pts2[1] * sizeof(float));
	int* ptrs = (int *) PyArray_GETPTR2(out_splists, 0, 0);
	memset((void *) ptrs, 0, size_pts3[0] * size_pts3[1] * sizeof(int));

	for(int h = 0; h < size_pts[0]; h++){
		Hypot& hypo = (*prev_hypo)[h];
		ptr2 = (float *) PyArray_GETPTR2(out2, h, 0);
		*ptr2++ = hypo.prob_acc;
		*ptr2++ = hypo.start_pos;
		*ptr2++ = hypo.end_pos;
		ptr = (int *) PyArray_GETPTR2(out, h, 0);
		for(int c = 0; c < hypo.text.size(); c++){
			*ptr++ = ((int) hypo.text[c]);
		}

		ptrs = (int *) PyArray_GETPTR2(out_splists, h, 0);
		for(size_t s = 0; s < hypo.splits.size(); s++){
			*ptrs++ = hypo.splits[s];
		}
	}

	PyObject* ret =  Py_BuildValue("(O, O, O)", out, out2, out_splists);
	Py_DECREF(out);
	Py_DECREF(out2);
	Py_DECREF(out_splists);

	return ret;
}

cv::RNG rng(12345);

int min_display_size = 200;

struct RenderedMser{

	cv::Rect bbox;

	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;

	std::set<int> regs_assigned;

	size_t mserId = -1;
};

static double distance_to_Line(cv::Point line_start, cv::Point line_end, cv::Point point)
{
	double normalLength = hypot(line_end.x - line_start.x, line_end.y - line_start.y);
	double distance = (double)((point.x - line_start.x) * (line_end.y - line_start.y) - (point.y - line_start.y) * (line_end.x - line_start.x)) / normalLength;
	return distance;
}

PyArrayObject* assign_lines(PyArrayObject* img, int numOfDims,  npy_intp* img_dims, PyArrayObject* det, npy_intp* det_dims)
{
	int type = CV_8UC3;
	if( numOfDims == 2 )
		type = CV_8UC1;

	cv::Mat srcImg = cv::Mat(img_dims[0], img_dims[1], type, PyArray_DATA(img) );
	float normFactor = sqrtf(srcImg.rows * srcImg.rows + srcImg.cols * srcImg.cols);

	cv::Mat detImg = cv::Mat(det_dims[0], det_dims[1], CV_32FC1, PyArray_DATA(det) );
	std::vector<cv::RotatedRect> rects;
	for(int i = 0; i < det_dims[0]; i++)
	{
		if (detImg.at<float>(i, 0) == 0 && detImg.at<float>(i, 1) == 0)
			continue;

		cv::RotatedRect rr( cv::Point(detImg.at<float>(i, 0) * img_dims[1], detImg.at<float>(i, 1) * img_dims[0]),
				cv::Size(detImg.at<float>(i, 2) * normFactor + detImg.at<float>(i, 3) * normFactor, detImg.at<float>(i, 3) * normFactor), detImg.at<float>(i, 4) * 180 / 3.14 );
		rects.push_back(rr);
	}

	std::map<int, int> line_to_rect;
	int next_line_id = 1;

	for( size_t i = 0; i < rects.size(); i++ ){
		cv::RotatedRect& rect1 = rects[i];

		for( size_t j = i + 1; j < rects.size(); j++ ){
			cv::RotatedRect& rect2 = rects[j];
			if( fabs(rect1.angle - rect2.angle) > 45.f / 6){
				continue;
			}

			float hr = std::min(rect1.size.height, rect2.size.height) / std::max(rect1.size.height, rect2.size.height);
			if(hr < 0.5)
				continue;

			std::vector<cv::Point2f> vertices;
			int ret = rotatedRectangleIntersection(rect1, rect2, vertices);
			if(ret != cv::INTERSECT_NONE){
#ifdef DEBUG
				cv::Mat draw = srcImg.clone();
				cv::rectangle(draw, rect1.boundingRect(), cv::Scalar(255, 0, 0));
#endif
				cv::Point2f pts[4];
				rect1.points(pts);

				cv::Point2f pts2[4];
				rect2.points(pts2);

				float height = std::min(rect1.size.height, rect2.size.height) / 3.0;

				double dist = fabs(distance_to_Line(pts[3], pts[0], pts2[0]));
				dist = std::max(dist, fabs(distance_to_Line(pts[3], pts[0], pts2[3])));
				if(fabs(dist) > height){
					//std::cout << "Height distance!\n";
					//cv::rectangle(draw, rect2.boundingRect(), cv::Scalar(255, 0, 0));
					//cv::imshow("noline", draw);
					//cv::waitKey(0);
					continue;
				}

				int line_id = -1;
				if(line_to_rect.find(i) != line_to_rect.end()){
					line_id = line_to_rect[i];
					line_to_rect[j] = line_id;
				}else if( line_to_rect.find(j) != line_to_rect.end() ){
					line_id = line_to_rect[j];
					line_to_rect[i] = line_id;
				}else{
					line_id = next_line_id++;
					line_to_rect[i] = line_id;
					line_to_rect[j] = line_id;
				}
			}
		}
	}

	for( size_t i = 0; i < rects.size(); i++ ){
		if(line_to_rect.find(i) != line_to_rect.end()){
			detImg.at<float>(i, 15) = line_to_rect[i];
		}
	}

	std::vector<cv::RotatedRect> line_rects;
	for(int lid = 1; lid < next_line_id; lid++){
		std::vector<cv::Point> cnt;
		for( size_t i = 0; i < rects.size(); i++ ){
			if(detImg.at<float>(i, 15) == lid){
				cv::RotatedRect& rect1 = rects[i];
				cv::Point2f pts[4];
				rect1.points(pts);
				cnt.push_back(pts[0]); cnt.push_back(pts[1]); cnt.push_back(pts[2]); cnt.push_back(pts[3]);
			}
		}
		if( cnt.size() < 3 ){
			//std::cout << "Bad line\n";
			continue;
		}

		cv::RotatedRect rect = cv::minAreaRect(cnt);
		if(rect.size.height > rect.size.width){
			std::swap(rect.size.height, rect.size.width);
			rect.angle += 90;
		}

		line_rects.push_back(rect);
	}

  npy_intp size_pts[2];
	size_pts[0] = line_rects.size();
	size_pts[1] = 5;

	PyArrayObject* out = (PyArrayObject *) PyArray_SimpleNew( 2, size_pts, NPY_FLOAT );
	//Py_INCREF(out);

	for(size_t i = 0; i < line_rects.size(); i++)
	{
		cv::RotatedRect& rect1 = line_rects[i];
		float* ptr = (float *) PyArray_GETPTR2(out, i, 0);
		*ptr++ = rect1.center.x;
		*ptr++ = rect1.center.y;
		*ptr++ = rect1.size.width;
		*ptr++ = rect1.size.height;
		*ptr++ = rect1.angle;
	}
	return out;

}

