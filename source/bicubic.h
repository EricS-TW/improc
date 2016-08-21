#pragma once
#include "Header.h"
#define PROGRAM_FILE "bicubic.cl"
#define KERNEL_FUNC "bicubic_mult"

Mat bicubic(Mat image);
float cubicInterpolate(float p[4], float x);
int bicubicInterpolate(float p[4][4], float alpha, float beta);