__kernel void bicubic_mult(__global unsigned char* src_R,
						  __global unsigned char* src_G,
						  __global unsigned char* src_B,
						  __global unsigned char* dst_R,
						  __global unsigned char* dst_G,
						  __global unsigned char* dst_B,						  
						  __global int* param)
{
	int index = get_global_id(0);

	int x_ratio = param[2];
	int y_ratio = param[3];
	
	int ori_width = param[1];
	int ori_height = param[0];
	
	int new_width = ori_width*x_ratio;
	int new_height = ori_height*y_ratio;
	
	int y = index / new_width;
	int x = index % new_width;

	float sample_R[4][4];
	float sample_G[4][4];
	float sample_B[4][4];

	int px = floor((float)x / x_ratio);
	int py = floor((float)y / y_ratio);

	dst_R[index] = 0;
	dst_G[index] = 0;
	dst_B[index] = 0;

	if (py<1 || px<1 || px >=(ori_width - 2) || py >=(ori_height - 2) || (x % x_ratio == 0 && y % y_ratio == 0))
	{
		dst_R[index] = src_R[py*ori_width + px];
		dst_G[index] = src_G[py*ori_width + px];
		dst_B[index] = src_B[py*ori_width + px];
	}
	else {
		for (int s = -1; s <= 2; s++)
			for (int t = -1; t <= 2; t++)
			{
				sample_R[s + 1][t + 1] = (float)src_R[(s + py)*ori_width + (t + px)];
				sample_G[s + 1][t + 1] = (float)src_G[(s + py)*ori_width + (t + px)];
				sample_B[s + 1][t + 1] = (float)src_B[(s + py)*ori_width + (t + px)];
			}
		float arr_R[4];
		float arr_G[4];
		float arr_B[4];
		float alpha = (float)((y%y_ratio)/ y_ratio);
		float beta = (float)((x%x_ratio) / x_ratio);
		for (int i = 0; i<4; i++) 
		{
			arr_R[i] = (float)(sample_R[i][1] + 0.5 * beta*(sample_R[i][2] - sample_R[i][0] + beta*(2.0*sample_R[i][0] - 5.0*sample_R[i][1] + 4.0*sample_R[i][2] - sample_R[i][3] + beta*(3.0*(sample_R[i][1] - sample_R[i][2]) + sample_R[i][3] - sample_R[i][0]))));
			arr_G[i] = (float)(sample_G[i][1] + 0.5 * beta*(sample_G[i][2] - sample_G[i][0] + beta*(2.0*sample_G[i][0] - 5.0*sample_G[i][1] + 4.0*sample_G[i][2] - sample_G[i][3] + beta*(3.0*(sample_G[i][1] - sample_G[i][2]) + sample_G[i][3] - sample_G[i][0]))));
			arr_B[i] = (float)(sample_B[i][1] + 0.5 * beta*(sample_B[i][2] - sample_B[i][0] + beta*(2.0*sample_B[i][0] - 5.0*sample_B[i][1] + 4.0*sample_B[i][2] - sample_B[i][3] + beta*(3.0*(sample_B[i][1] - sample_B[i][2]) + sample_B[i][3] - sample_B[i][0]))));
		}
		dst_R[index] = (unsigned char)(floor((float)(arr_R[1] + 0.5 * alpha*(arr_R[2] - arr_R[0] + alpha*(2.0*arr_R[0] - 5.0*arr_R[1] + 4.0*arr_R[2] - arr_R[3] + alpha*(3.0*(arr_R[1] - arr_R[2]) + arr_R[3] - arr_R[0]))))));
		dst_G[index] = (unsigned char)(floor((float)(arr_G[1] + 0.5 * alpha*(arr_G[2] - arr_G[0] + alpha*(2.0*arr_G[0] - 5.0*arr_G[1] + 4.0*arr_G[2] - arr_G[3] + alpha*(3.0*(arr_G[1] - arr_G[2]) + arr_G[3] - arr_G[0]))))));
		dst_B[index] = (unsigned char)(floor((float)(arr_B[1] + 0.5 * alpha*(arr_B[2] - arr_B[0] + alpha*(2.0*arr_B[0] - 5.0*arr_B[1] + 4.0*arr_B[2] - arr_B[3] + alpha*(3.0*(arr_B[1] - arr_B[2]) + arr_B[3] - arr_B[0]))))));
	}
}
