#pragma once

#include "config.h"

// 计算x([-255, 255])绝对值, 计算公式：abs(a) = X[a+255]
void calcuXabs();

// 计算一副灰度图像的梯度值，范围：[0, 255]
double BlurDetect(const Mat& img);
