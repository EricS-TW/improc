#pragma once
#include "Header.h"
#define pi 3.141592
#define sigma 0.84089642
#define PROGRAM_FILE "blur.cl"
#define KERNEL_FUNC "blur_mult"

Mat gaussian_blur(Mat image);