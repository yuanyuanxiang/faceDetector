
#include "../pyCaller.h"
#include "../CvxText.h"

// 测试：对"image.jpg"进行人脸检测
int main()
{
	cv::Mat m = imread("image.jpg");
	if (false == m.empty())
	{
		pyCaller py;
		py.SetPythonHome("D:/Anaconda3/envs/tfgpu");
		py.Init("detect");
		py.ActivateFunc("test_src");
		npy_intp dims[] = { m.rows, m.cols, 3 }; // 给定维度信息
		// 生成包含这个多维数组的PyObject对象，使用PyArray_SimpleNewFromData函数
		// 第一个参数2表示维度，第二个为维度数组Dims,第三个参数指出数组的类型，第四个参数为数组
		PyObject *PyArray  = PyArray_SimpleNewFromData(3, dims, NPY_UBYTE, m.data);
		// 同样定义大小与Python函数参数个数一致的PyTuple对象
		PyObject *ArgArray = PyTuple_New(1);
		PyTuple_SetItem(ArgArray, 0, PyArray); 
		tfOutput tf(100);
		py.CallFunction("test_src", ArgArray, &tf);
		if (tf.output())
		{
			const float *p = tf.boxes;
			float x1 = *p++, y1 = *p++, x2 = *p++, y2 = *p++;
			cv::Rect rect(CvPoint(x1, y1), CvPoint(x2, y2));
			cv::rectangle(m, rect, CV_RGB(0, 0, 255), 2);
			CvxText font("simsun.ttc");
			char text[256];
			sprintf_s(text, "%s:%.3f", "人脸", *p);
			font.putText(m, text, cv::Point(rect.x, rect.y), 
				CV_RGB(0, 255, 0));
			imshow("face", m);
			waitKey(0);
		}
	}
	system("PAUSE");
	return 0;
}
