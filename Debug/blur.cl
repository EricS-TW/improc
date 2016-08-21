__kernel void blur_mult(__global unsigned char* src_R,
						  __global unsigned char* src_G,
						  __global unsigned char* src_B,
						  __global unsigned char* dst_R,
						  __global unsigned char* dst_G,
						  __global unsigned char* dst_B,
						  __global float* Filter,
						  __global int* param)
{ 
	int index = get_global_id(0);
	
	int width = param[0];
	int height = param[1];
	int size = param[2];

	int i = index / width;
	int j = index % width;
	int start = (size >> 1) -1;

	if( /*i > start  && i < height - start &&
		j > start  && j < width - start*/1)
	{
		dst_R[ index ] = 0;
		dst_G[ index ] = 0;
		dst_B[ index ] = 0;

		for(int m = 0; m < size; m++)
			for(int n = 0; n < size; n++)
			{
				dst_R[ index ] += src_R[ (i - (size >> 1) + m)*width + j - (size >> 1) + n ] * Filter[ m*size + n];
				dst_G[ index ] += src_G[ (i - (size >> 1) + m)*width + j - (size >> 1) + n ] * Filter[ m*size + n];
				dst_B[ index ] += src_B[ (i - (size >> 1) + m)*width + j - (size >> 1) + n ] * Filter[ m*size + n];
			}				
	}
	else
	{
		dst_R[ index ] = src_R[ index ];
		dst_G[ index ] = src_G[ index ];
		dst_B[ index ] = src_B[ index ];
	}
}