#include "bicubic.h"

float cubicInterpolate(float p[4], float x) {
	return (float)(p[1] + 0.5 * x*(p[2] - p[0] + x*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x*(3.0*(p[1] - p[2]) + p[3] - p[0]))));
};

int bicubicInterpolate(float p[4][4], float alpha, float beta) {
	float arr[4];
	arr[0] = cubicInterpolate(p[0], beta);
	arr[1] = cubicInterpolate(p[1], beta);
	arr[2] = cubicInterpolate(p[2], beta);
	arr[3] = cubicInterpolate(p[3], beta);
	return (int)(floor(cubicInterpolate(arr, alpha)));
};

Mat bicubic(Mat image)
{
	// input ratio value
	int x_ratio, y_ratio;
	x_ratio = 1; y_ratio = 1;
	cout << "* Enter the x-ratio value (x>0):";  
	cin >> x_ratio;cout << endl;
	cout << "* Enter the y-ratio value (y>0):" ; 
	cin >> y_ratio;cout << endl;

	if (x_ratio < 0 || y_ratio < 0) {
		cout << "ratio couldn't smaller than 0" << endl;
		exit(-1);
	}
	int param[4];
	param[0] = image.rows;
	param[1] = image.cols;
	param[2] = x_ratio;
	param[3] = y_ratio;
	/* Host/device data structures */
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_int err;

	/* Program/kernel data structures */
	cl_program program;
	FILE *program_handle;
	char *program_buffer, *program_log;
	size_t program_size, log_size;
	cl_kernel kernel;

	/*buffers */
	cl_mem src_R_buff, src_G_buff, src_B_buff;
	cl_mem dst_R_buff, dst_G_buff, dst_B_buff;
	cl_mem Filter_buff;
	cl_mem param_buff;
	size_t work_units_per_kernel;


	unsigned char *src_R = new unsigned char[image.cols*image.rows];
	unsigned char *src_G = new unsigned char[image.cols*image.rows];
	unsigned char *src_B = new unsigned char[image.cols*image.rows];

	unsigned char *dst_R = new unsigned char[image.cols*image.rows*x_ratio*y_ratio];
	unsigned char *dst_G = new unsigned char[image.cols*image.rows*x_ratio*y_ratio];
	unsigned char *dst_B = new unsigned char[image.cols*image.rows*x_ratio*y_ratio];


	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++)
		{
			src_B[i*image.cols + j] = image.at<Vec3b>(i, j)[0];
			src_G[i*image.cols + j] = image.at<Vec3b>(i, j)[1];
			src_R[i*image.cols + j] = image.at<Vec3b>(i, j)[2];
		}

	/* Identify a platform */
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err < 0) {
		perror("Couldn't find any platforms");
		exit(-1);
	}

	/* Access a device */
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1,
		&device, NULL);
	if (err < 0) {
		perror("Couldn't find any devices");
		exit(-1);
	}

	/* Create the context */
	context = clCreateContext(NULL, 1, &device, NULL,
		NULL, &err);
	if (err < 0) {
		perror("Couldn't create a context");
		exit(-1);
	}

	/* Read program file and place content into buffer */
	program_handle = fopen(PROGRAM_FILE, "r");
	if (program_handle == NULL) {
		perror("Couldn't find the program file");
		exit(-1);
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	/* Create program from file */
	program = clCreateProgramWithSource(context, 1,
		(const char**)&program_buffer, &program_size, &err);
	if (err < 0) {
		perror("Couldn't create the program");
		exit(-1);
	}
	free(program_buffer);

	/* Build program */
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		exit(-1);
	}

	/* Create kernel for the mat_vec_mult function */
	kernel = clCreateKernel(program, KERNEL_FUNC, &err);
	if (err < 0) {
		perror("Couldn't create the kernel");
		exit(-1);
	}

	/* Create CL buffers to hold input and output data */
	src_R_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, sizeof(unsigned char)* image.cols*image.rows, src_R, &err);
	src_G_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, sizeof(unsigned char)* image.cols*image.rows, src_G, &err);
	src_B_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, sizeof(unsigned char)* image.cols*image.rows, src_B, &err);
	param_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, sizeof(int) * 4, param, &err);

	if (err < 0) {
		perror("Couldn't create a buffer object");
		exit(-1);
	}
	dst_R_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		sizeof(unsigned char)* image.cols*image.rows*x_ratio*y_ratio, NULL, NULL);
	dst_G_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		sizeof(unsigned char)* image.cols*image.rows*x_ratio*y_ratio, NULL, NULL);
	dst_B_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		sizeof(unsigned char)* image.cols*image.rows*x_ratio*y_ratio, NULL, NULL);

	/* Create kernel arguments from the CL buffers */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_R_buff);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &src_G_buff);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &src_B_buff);
	if (err < 0) {
		perror("Couldn't set the kernel argument");
		exit(-1);
	}
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &dst_R_buff);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &dst_G_buff);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &dst_B_buff);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), &param_buff);

	/* Create a CL command queue for the device*/
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	if (err < 0) {
		perror("Couldn't create the command queue");
		exit(-1);
	}

	/* Enqueue the command queue to the device */
	work_units_per_kernel = image.cols*image.rows*x_ratio*y_ratio; /* 4 work-units per kernel */

	cl_event event;
	cl_ulong time_start;
	cl_ulong time_end;

	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_units_per_kernel,
		NULL, 0, NULL, &event);
	clWaitForEvents(1, &event);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	printf("GPU : %0.3f  ms\n", ((time_end - time_start) / 1000000.0));

	if (err < 0) {
		perror("Couldn't enqueue the kernel execution command");
		exit(-1);
	}

	/* Read the result */
	err = clEnqueueReadBuffer(queue, dst_R_buff, CL_TRUE, 0, sizeof(unsigned char)* image.cols*image.rows*x_ratio*y_ratio,
		dst_R, 0, NULL, NULL);
	err = clEnqueueReadBuffer(queue, dst_G_buff, CL_TRUE, 0, sizeof(unsigned char)* image.cols*image.rows*x_ratio*y_ratio,
		dst_G, 0, NULL, NULL);
	err = clEnqueueReadBuffer(queue, dst_B_buff, CL_TRUE, 0, sizeof(unsigned char)* image.cols*image.rows*x_ratio*y_ratio,
		dst_B, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't enqueue the read buffer command");
		exit(-1);
	}
	
	Mat new_image = Mat::zeros(image.rows*y_ratio, image.cols*x_ratio, image.type());

	clock_t start_time, end_time;
	float total_time = 0;	
	start_time = clock();
	uchar tmp = 0;
	for (int y = 0; y < new_image.rows; y++)
	{
		for (int x = 0; x < new_image.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				float sample_matrix[4][4];
				int px = (int)(floor(x / x_ratio));
				int py = (int)(floor(y / y_ratio));

				if (py<1 || px<1 || px >=(image.cols - 2) || py >=(image.rows - 2) || (x % x_ratio == 0 && y % y_ratio == 0))
				{
					tmp = image.at<Vec3b>(py, px)[c];
				}
				else {
					for (int s = -1; s <= 2; s++)
					{
						for (int t = -1; t <= 2; t++)
						{
							//if (s + py<0 || s + py>=image.rows || t + px<0 || t + px>=image.cols) {
							//	cout << "px = " << px << " py =" << py << endl;
							//}
							sample_matrix[s + 1][t + 1] = (float)(image.at<Vec3b>(s + py, t + px)[c]);
						}
					}					
					tmp = (uchar)(bicubicInterpolate(sample_matrix,(float)((y%y_ratio) / y_ratio), (float)((x%(x_ratio)) / x_ratio)));
				}
			}
		}
	}
	end_time = clock();
	total_time = (float)(end_time - start_time);
	printf("CPU : %0.3f ms \n", total_time);
	for (int i = 0; i < new_image.rows; i++)
		for (int j = 0; j < new_image.cols; j++)
		{
			new_image.at<Vec3b>(i, j)[0] = dst_B[i*new_image.cols + j];
			new_image.at<Vec3b>(i, j)[1] = dst_G[i*new_image.cols + j];
			new_image.at<Vec3b>(i, j)[2] = dst_R[i*new_image.cols + j];
		}

	delete[] src_R;
	delete[] src_G;
	delete[] src_B;
	delete[] dst_R;
	delete[] dst_G;
	delete[] dst_B;

	/*
	/// Create Windows
	namedWindow("Original Image", 1);
	namedWindow("New Image", 1);

	/// Show stuff
	imshow("Original Image", image);
	imshow("New Image", new_image);

	waitKey(1);
	*/

	/* Deallocate resources */
	clReleaseMemObject(src_R_buff);
	clReleaseMemObject(src_G_buff);
	clReleaseMemObject(src_B_buff);
	clReleaseMemObject(dst_R_buff);
	clReleaseMemObject(dst_G_buff);
	clReleaseMemObject(dst_B_buff);
	clReleaseMemObject(param_buff);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseEvent(event);
	image.release();

	return new_image;
};