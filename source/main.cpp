#include "Header.h"
#include "bicubic.h"
#include "Gaussian_Blur.h"

int main(int argc,char *argv[]){
	//Read image
	Mat image;
	string filename;
	if( argc >2 ){
		image = imread(argv[argc-1]);
		filename = string(argv[argc - 1]);

		if(image.empty()){
			cout << "couldn't open image file";
		}
	
		//Imagproc 
		for (int i = 1; i < argc-1 ; i++) {
			string arg = string(argv[i]);
			if (arg == "-g") {
				image = gaussian_blur(image);
				filename.insert(filename.find('.'), "_blur");
			}
			else if (arg == "-b") {
				image = bicubic(image);
				filename.insert(filename.find('.'), "_bicubic");
			}
			else {
				cout << "couldn't find the right command" << endl;
				cout << arg << endl;
			}
		}
		imwrite(filename.c_str(),image);
		return 0;
	}
	else {
		cout << "not enough command arguments";
		return -1;
	}
	
}