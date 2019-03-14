#include "blurDetect.h"

static int X_abs[512];


void calcuXabs()
{
	for (int i = 0; i < 256; ++i)
	{
		// x = -255, -254, -253, ..., 0
		X_abs[i] = 255 - i;
		// x = 0, 1, 2, ..., 255
		X_abs[255 + i] = i;
	}
}


double BlurDetect(const Mat& img)
{
	const int w_s = 16; // 宽度开始点
	const int h_s = 16; // 高度开始点
	const int w_e = img.cols - w_s; // 宽度结束点
	const int h_e = img.rows - h_s; // 高度结束点

	const int channel = img.channels();
	const int rowlen = img.step[0];
	const uchar* pst1 = img.data + (h_s-1) * rowlen;
	const uchar* pst2 = img.data + h_s     * rowlen;
	const uchar* pst3 = img.data + (h_s+1) * rowlen;

	int sum_all = 0;
	for (int i = h_s; i < h_e; i += 2, pst1 += 2*rowlen, pst2 += 2*rowlen, pst3 += 2*rowlen)
	{
		for (int j = w_s, offset = j * channel; j < w_e; j += 2, offset += 2*channel)
		{
			const uchar *p1 = pst1 + offset, *p2 = pst2 + offset, *p3 = pst3 + offset, t = 255 - *p2;
			sum_all += X_abs[*(p1 - channel) + t] + X_abs[*p1 + t] + X_abs[*(p1 + channel) + t];
			sum_all += X_abs[*(p2 - channel) + t]                  + X_abs[*(p2 + channel) + t];
			sum_all += X_abs[*(p3 - channel) + t] + X_abs[*p3 + t] + X_abs[*(p3 + channel) + t];
		}
	}
	int size = 2 * (h_e-h_s) * (w_e-w_s);
	double ret = (double)sum_all / size;

	return ret;
}
