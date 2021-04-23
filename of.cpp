//*************************************************************************
// Example program using OpenCV library
//					
// @file	of.cpp
// @author Luis M. Jiménez
// @date 2021
//
// @brief Course: Computer Vision (1782)
// Dpo. of Systems Engineering and Automation
// Automation, Robotics and Computer Vision Lab (ARVC)
// http://arvc.umh.es
// University Miguel Hernández
//
// @note Description: 
//	- Shows the calculation of optical Flow. Algorithms:
//		LK  Lucas-Kanade + GFFT algorithm [Bouguet00]
//		FB  Gunnar Farneback's algorithm [Farneback03], 
//		DT  DualTV-L1 [Zach07-Javier2012]  
//		CS  CAMShift [Bradski98]
//		KF  Kalman Filter - GFFT,  (Not working yet) 
//		BS  Background Supression MOG2 [Zivkovic2004]
//
//	- Captures images from a file or a camera
//
//  Dependencies:
//      OpenCV 3.4.12 or higher 
//
//*************************************************************************
#define _USE_MATH_DEFINES

#include <cassert>	// debugging
#include <iostream>	// c++ standar input/output
#include <sstream>  	// string to number conversion 
#include <string>	// handling strings
#include <vector>	// handling vectors
#include <math.h>	// math functions
#include <stdio.h>	// standar C input/ouput

#include <opencv2/opencv.hpp>  // OpenCV library headers

//*************************************************************************
// C++ namespaces C++ (avoids using prefix for standard classes)
// for OpenCV  use prefix    cv::
//*************************************************************************
using namespace std;


//*************************************************************************
// Function Prototypes
//*************************************************************************
int LucasKanadeOF(cv::VideoCapture &camera, vector<int> exit_keys =  vector<int>() );
int FarnebackOF(cv::VideoCapture &camera,  vector<int> exit_keys =  vector<int>());
int DualTVL1OF(cv::VideoCapture &camera,  vector<int> exit_keys =  vector<int>());
int CAMShiftOF(cv::VideoCapture &camera,  vector<int> exit_keys =  vector<int>());
int KalmanFilterOF(cv::VideoCapture &camera,  vector<int> exit_keys =  vector<int>());
int BackgroundSuppressionMOG(cv::VideoCapture &camera,  vector<int> exit_keys =  vector<int>());

int CalcHistWindowCAMShift(cv::Mat &frame, cv::Mat &roi_hist, cv::Mat &display_hist, int histMode);
static inline cv::Point calcPoint(cv::Point2f center, double R, double angle);

void ShowOpticalFlowHue(cv::Mat &flow, cv::Mat &disp);
void ShowOpticalFlowArrow(cv::Mat &flow, cv::Mat &disp);
void ShowOpticalFlowArrow(vector<cv::Point2f> &p0, vector<cv::Point2f> &p1, vector<uchar> status, cv::Mat &disp);
void ShowHistogram(cv::Mat &hist, cv::Mat &disp, bool hueColor=false);

static void onMouse( int event, int x, int y, int, void* ptData );

//*************************************************************************
// Constants 
//*************************************************************************
const char  * WINDOW_CAMERA1  = "(W1) Camera 1";			// windows id
const char  * WINDOW_HISTOGRAM1  = "(W2) Histogram";		// windows id

const char  * WINDOW_PROCESSING1  = "(W2) Processing ";		// windows id
const char  * WINDOW_PROCESSING2  = "(W3) Processing";		// windows id


//*************************************************************************
// Global Variables
//*************************************************************************
int CAMERA_ID = 0;	//  default camera
cv::Size camSize(800, 600);

char *IMAGE_FILE = NULL;

int DELAY = 1000;		// ms diplay time when a file is processed

bool HELP = false;

string TYPE = "FB";		// LK (Lucas-Kanade Bouguet0]), FB (Gunnar Farneback's algorithm  Farneback03)
						// DT (DualTV-L1 Zach07-Javier2012),  CS (CAMShift Bradski98)
						// KF (Kalman Filter- GFFT), BS (Background Supression MOG Zivkovic2004) 

bool SELECTED_OBJECT = false;
cv::Rect SELECTION;

//*************************************************************************
// Functions
//*************************************************************************

int main (int argc, char * const argv[]) 
{
	cv::VideoCapture camera;	// Cameras	
    
	//-------------------------------------------
	// Put here the code to Intialize objets 
	//-------------------------------------------

	// locale::global(locale("spanish"));	// Character set (so that you can use accents)

     
    // check command line parameters
    for(int i=1; i< argc; i++)
    {
        if (string(argv[i])=="-c" && (i+1)<argc  && argv[i+1][0]!='-') { CAMERA_ID =atoi(argv[i+1]); i++; }

		else if(string(argv[i])=="-t" && (i+1)<argc  && argv[i+1][0]!='-') { TYPE = argv[i+1]; i++; }
                 
     	else if (string(argv[i])=="-w" && (i+1)<argc  && argv[i+1][0]!='-') { DELAY =atoi(argv[i+1]); i++; }
 
        else if(string(argv[i])=="-h" || string(argv[i])=="--help") { HELP = true;}   
        
        else { IMAGE_FILE = argv[i];}	// the image video file
    }
    
    if(HELP)
    {
    	 cout << argv[0] << "   -c Camera_ID   [ImageVideoFile]\n" << endl;
    	
    	 cout << " [ImageVideoFile] is OPTIONAL. If no Image File is passed as parameter, a camera will be used. " << endl;
    	 cout << "	-t <Type>   => LK (Lukas kanade - GFTT), FB (Farneback Dense OF)" << endl;
		 cout << "                 DT (DualTV-L1 Zach2007-Javier2012)" << endl;
		 cout << "                 CS (CAMShift Bradski98),  KF (Kalman Filter - GFFT)" << endl;
		 cout << "                 BS (Background Supression MOG [Zivkovic2004])" << endl;
   		 cout << "	-w <delay>   => file mode display time in ms (0 until exit) (def: 5000ms)" << endl;
	  	 cout << " -h, --help  => shows help\n" << endl;
            
    	 return 0;
    }
    

	// Configuring cameras or Image video file
	if(IMAGE_FILE)
	{
		cout << "Opening video file: " << IMAGE_FILE << endl;
		camera.open(IMAGE_FILE);
		if (!camera.isOpened())
		{
			cout << "-- unnable to open de video file: " << IMAGE_FILE << ", sorry.\n";
			return -1;
		}
	}
	else
	{
		cout << "Opening camera: " << CAMERA_ID << endl;
		camera.open(CAMERA_ID); //  open  camera
		if (!camera.isOpened())
		{
			cout << "-- unnable to open camera " << CAMERA_ID << ", sorry.\n";
			return -1;
		}
		//setting capture properties (low resolution to speed up detection)
		//camera.set(CV_CAP_PROP_FRAME_WIDTH, camSize.width);
		//camera.set(CV_CAP_PROP_FRAME_HEIGHT, camSize.height);

		// Getting camera resolution
		camSize.width = (int) camera.get(CV_CAP_PROP_FRAME_WIDTH);
		camSize.height = (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT);
	}

  	cout << "  ----------------------------------------------------------------------" << endl;
	cout << "  (q,Q,Esc) Exit\n" << endl;
	cout << "  Select Optical Flow type: " << endl;
	cout << "       (l,L) Lucas-Kanade - GFTT Optical Flow [Bouguet2000]" << endl;
	cout << "       (f,F) Farneback Dense Optical Flow [Farneback2003]" << endl;
 	cout << "       (d,D) DualTV-L1 Dense Optical Flow [Zach2007-Javier2012]" << endl;
 	cout << "       (m,M) CAMShift - Mean-Shift Tracking [Bradski98]" << endl;
	cout << "       (k,K) Kalman Filter - GFTT Optical Flow" << endl;
	cout << "       (b,B) Backbrougnd Supression MOG2 [Zivkovic2004]" << endl;
  	cout << "  ----------------------------------------------------------------------" << endl;
 	cout << "  (r,R) Reset reference points in sparse optical flow (LK)" << endl;
 	cout << "  (a,A) Show arrows for dense/sparse optical flow" << endl;
 	cout << "  (c,C) Show Hue color for dense optical flow (Angle:Hue, Mod:Intensity)" << endl;
	cout << "  (t,T) Show Tracking for sparse optical flow (LK)" << endl;
  	cout << "  ----------------------------------------------------------------------" << endl;
	cout << "  CAMShift (Continuos Adaptative Mean-Shift) tracking algorithm: " << endl;
 	cout << "       (p,P) Select Center Point (fixed window Size)" << endl;
	cout << "       (w,W) Mouse Left-Click-and-Drag to select Window" << endl;
	cout << "       (r,R) Clear Selection Window" << endl;
	cout << "       (h,H) Select Hue Histogram (clear window)" << endl;
	cout << "       (i,I) Select Intensity Histogram (clear window)" << endl;
	cout << "       (1) CAMShift (Countinuous Adaptative Mean-Shift) [Zivkovic2006]" << endl;
	cout << "       (2) Mean Shift Algorithm" << endl;
	cout << "  ----------------------------------------------------------------------" << endl;
	cout << "  Background Supression - Mixture of Gaussians: " << endl;
	cout << "       (1) KNN Algorithm [Zivkovic2006]" << endl;
	cout << "       (2) MOG2 Algorithm [Zivkovic2004]" << endl; 
	cout << "       (c,C) Process color image" << endl;
	cout << "       (i,I) Process Intensity (gray) image" << endl;
 	cout << "       (r,R) Reset bakcground model" << endl;
 	cout << "  ----------------------------------------------------------------------" << endl;


    //-------------------------------------------
    // Main processing loop
    //-------------------------------------------
	int arr[] = { 'q', 'Q', 27, 'f', 'F', 'l', 'L', 'd', 'D', 'm', 'M', 'b', 'B', 'k', 'K'};
	vector<int> EXIT_KEYS(arr, arr + sizeof(arr) / sizeof(arr[0]));

	int key = 1;
	while (key>0)
	{
		if( TYPE == "FB")
			key = FarnebackOF(camera, EXIT_KEYS);
		else if( TYPE == "DT")
			key = DualTVL1OF(camera, EXIT_KEYS);
		else if( TYPE == "CS")
			key = CAMShiftOF(camera, EXIT_KEYS);
		else if( TYPE == "BS")
			key = BackgroundSuppressionMOG(camera, EXIT_KEYS);
		//else if( TYPE == "KF")
		//	key = KalmanFilterOF(camera, EXIT_KEYS);  	// Not working yet
		else
			key = LucasKanadeOF(camera, EXIT_KEYS);
	

        if (key == 'q' || key == 'Q' || key == 27)  break;
		else if (key == 'f' || key == 'F' )		TYPE = "FB";
		else if (key == 'l' || key == 'L' )		TYPE = "LK";
		else if (key == 'd' || key == 'D' )		TYPE = "DT";
		else if (key == 'm' || key == 'M' )		TYPE = "CS";
		else if (key == 'k' || key == 'K' )		TYPE = "KF";
		else if (key == 'b' || key == 'B' )		TYPE = "BS";

		cv::destroyAllWindows();
	}

   
    //-------------------------------------------
	// free windows and camera resources
	//-------------------------------------------
	cv::destroyAllWindows();
	if (camera.isOpened())	camera.release();
	
	// programm exits with no errors
	return 0;
}


//----------------------------------------------------------------------
// Visualize an Histogram in color
//----------------------------------------------------------------------
// hist: Histogram
// disp: Display image
//----------------------------------------------------------------------
void ShowHistogram(cv::Mat &hist, cv::Mat &disp, bool hueColor)
{
		if(hist.empty()) return;

		int hsize = hist.rows;

		//cout << "Hist size:" << hsize << endl;

		if(!disp.empty())
			disp.release();
				
		int disp_width = hsize;
		if(hsize<300) disp_width = hsize*2;

		int disp_height = cvRound(disp_width*0.6f);	// window aspect ratio
		int disp_top_height =  cvRound(disp_height*0.95f);	// useful diplay height 95%
		int disp_bottom = cvRound(disp_height*0.975f);		// bottom of the display area of the histogram 2.5% for the botton area
	
   		double maxVal;
		cv::minMaxIdx(hist, NULL, &maxVal, NULL, NULL);	// maximum value of the histogram

		double scaleH=1.0;
		if(maxVal>0) scaleH = (double)disp_top_height/maxVal;	//  vertical scale

		int binW = cvRound(disp_width / hsize);	//  horizontal scale (integer)
	
		disp = cv::Mat::zeros(disp_height, disp_width, CV_8UC3);
		disp = cv::Scalar::all(255);

		// Hue colors for each histogram bin
		cv::Mat buf(1, hsize, CV_8UC3);
		if(hueColor)
		{
			for( int i = 0; i < hsize; i++ )
				buf.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180./hsize), 255, 255);
			cvtColor(buf, buf, CV_HSV2BGR);
		}
		else  buf = cv::Scalar(150,100,100);	// fixed color
			
		// draws the histogram
        for( int i = 0; i < hsize; i++ )
        {
            int val = cv::saturate_cast<int>(hist.at<float>(i)*scaleH);
            cv::rectangle( disp, cv::Point(i*binW, disp_bottom), cv::Point((i+1)*binW, disp_bottom - val),
                        cv::Scalar(buf.at<cv::Vec3b>(i)), -1, 8 );
        }

}


//----------------------------------------------------------------------
// shows Optical Flow as a Hue color coded image
//----------------------------------------------------------------------
// flow: Optical Flow image (two channels [0] -> vx,  [0] -> vy)  CV_32FC2
// disp: display image
//----------------------------------------------------------------------
void ShowOpticalFlowHue(cv::Mat &flow, cv::Mat &disp)
{
	// visualization
    cv::Mat flow_parts[2];
    cv::split(flow, flow_parts);
        
	cv::Mat magnitude, angle, magn_norm, angle_norm;
    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, false);		// radians
    cv::normalize(magnitude, magn_norm, 0.0f, 255.0f, cv::NORM_MINMAX); // normalize magnitude -> [0-255.0]
    angle_norm = angle * (180.f / (float)M_PI /2 );	// normalize angle [0-2*pi] -> [0-180.0]  (angle/2)
 
	//build hsv image (optical flow color depending of the angle and magnitude)
    cv::Mat _hsv[3], hsv, hsv8, colors;
    _hsv[0] = angle_norm;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F)*255.f;
    _hsv[2] = magn_norm;
    cv::merge(_hsv, 3, hsv);       
	hsv.convertTo(hsv8, CV_8U);
    cvtColor(hsv8, disp, cv::COLOR_HSV2BGR);

}


//----------------------------------------------------------------------
// shows Optical Flow as colored arrows (arrow colors depending of the angle and magnitude)
//----------------------------------------------------------------------
// flow: Optical Flow image (two channels [0] -> vx,  [0] -> vy)  CV_32FC2
// disp: display image (arrows are overlayed  over the previous content - color image)
//----------------------------------------------------------------------
void ShowOpticalFlowArrow(cv::Mat &flow, cv::Mat &disp)
{
	// visualization

    cv::Mat flow_parts[2];
    cv::split(flow, flow_parts);
        
	cv::Mat magnitude, angle, magn_norm, angle_norm;
    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, false);		// radians
    cv::normalize(magnitude, magn_norm, 0.0f, 255.0f, cv::NORM_MINMAX); // normalize magnitude -> [0-255.0]
    angle_norm = angle * (180.f / (float)M_PI /2 );	// normalize angle [0-2*pi] -> [0-180.0]  (angle/2)
 
	//build hsv image (arrow colors depending of the angle and magnitude)
    cv::Mat _hsv[3], hsv, hsv8, colors;
    _hsv[0] = angle_norm;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F)*255.f;
    _hsv[2] = magn_norm;
    cv::merge(_hsv, 3, hsv);       
	hsv.convertTo(hsv8, CV_8U);
    cvtColor(hsv8, colors, cv::COLOR_HSV2BGR);


	// checks that disp is a color image of same size that flow
	if(disp.empty() || disp.cols!=flow.cols || disp.rows!=flow.rows)
		disp = cv::Mat(flow.rows, flow.cols, CV_8UC3);
	else if(disp.type() != CV_8UC3)
		cvtColor(disp, disp, cv::COLOR_GRAY2BGR);

	cv::Scalar color = cv::Scalar(0,0,255);
	float da = (float)(30.0*M_PI/180.0);	// arrow apperture	
	float dm= 0.20f;						// arrow lenght (percentage)
	float min_of = 2.0f;	// minimum legnth of optical flow to be visualized

	int window = 11; 	// Optical flow window sampling size (must be an odd number)
	for(int f=0; f< flow.rows; f+=window)
		for(int c=0; c< flow.cols; c+=window)
		{
			int fm=f, cm=c;		// local maxima

			if(c>window/2 && f>window/2 && c<flow.cols-window/2 && f<flow.rows-window/2)
			{
				// select the local maxima in the sampled ROI
				cv::Mat roi_m = magnitude(cv::Rect(c-window/2, f-window/2, window, window));	// local ROI
				cv::Point max_loc;		
				cv::minMaxLoc(roi_m, NULL, NULL, NULL, &max_loc);	// find maximum location
				fm = max_loc.y + f-window/2;  cm = max_loc.x + c-window/2;
			}

			float vx = flow_parts[0].at<float>(fm,cm);
			float vy = flow_parts[1].at<float>(fm,cm);
			float m = magnitude.at<float>(fm,cm);
			float a = angle.at<float>(fm,cm);
			color = (cv::Scalar) colors.at<cv::Vec3b>(fm,cm);

			if(m < min_of)	continue;	// minimum legnth of optical flow to be visualized

			cv::Point2i p1 = cv::Point2i(cm, fm);
			cv::Point2i p2 = cv::Point2i(cvRound(cm+vx), cvRound(fm+vy));
			if(p2.x>=flow.cols  || p2.y>=flow.rows)
				continue;	// arrow outside of image
				
			// arrow
			cv::Point2i p3 = p2 - cv::Point(cvRound(m*dm*cos(a-da)),cvRound(m*dm*sin(a-da)));
			cv::Point2i p4 = p2 - cv::Point(cvRound(m*dm*cos(a+da)),cvRound(m*dm*sin(a+da)));
	
			cv::line(disp, p1, p2, color, 1);
			cv::line(disp, p2, p3, color, 1);
			cv::line(disp, p2, p4, color, 1);
		}

}


//----------------------------------------------------------------------
// shows Sparse Optical Flow as colored arrows (arrow colors depending of the angle and magnitude)
//----------------------------------------------------------------------
// p0: points previous location
// p1: points currentus location
// disp: display image (arrows are overlayed  over the previous content - color image)
//----------------------------------------------------------------------
void ShowOpticalFlowArrow(vector<cv::Point2f> &p0, vector<cv::Point2f> &p1, vector<uchar> status, cv::Mat &disp)
{
	// visualization
    
	// Create some random colors
    vector<cv::Scalar> colors;
    cv::RNG rng;
    for(int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(cv::Scalar(r,g,b));
    }

	// checks that disp is a color image 
	if(disp.type() != CV_8UC3)
		cvtColor(disp, disp, cv::COLOR_GRAY2BGR);

	float da = (float)(30.0*M_PI/180.0);	// arrow apperture	
	float dm= 0.20f;						// arrow lenght (percentage)
	float min_of = 2.0f;	// minimum legnth of optical flow to be visualized

	int window = 11; 	// Optical flow window sampling size (must be an odd number)
	for(unsigned int i=0; i< p0.size(); i++)
	{
		// Draw good points
        if(status[i] == 1) {
			float m = sqrt((p1[i].x-p0[i].x)*(p1[i].x-p0[i].x)+ (p1[i].y-p0[i].y)*(p1[i].y-p0[i].y));
			float a = atan2(p1[i].y-p0[i].y,p1[i].x-p0[i].x );

			if(m < min_of)	{
				cv::circle(disp, p1[i], 4, colors[i], 2);
				continue;	// minimum legnth of optical flow to be visualized
			}

			// arrow
			cv::Point2i p3 = p1[i] - cv::Point2f(m*dm*cos(a-da), m*dm*sin(a-da));
			cv::Point2i p4 = p1[i] - cv::Point2f(m*dm*cos(a+da), m*dm*sin(a+da));

			cv::circle(disp, p0[i], 4, colors[i], 2);
			cv::line(disp, p0[i], p1[i], colors[i], 1);
			cv::line(disp, p1[i], p3, colors[i], 1);
			cv::line(disp, p1[i], p4, colors[i], 1);
		}
	}

}

//----------------------------------------------------------------------
// computes Optical Flow Lucas-Kanade Algorithm [Bouguet2000] Iteartive implementation
//----------------------------------------------------------------------
// camera: Camera/video file to process
// exit_keys: exit keys (optional)
//----------------------------------------------------------------------
int LucasKanadeOF(cv::VideoCapture &camera, vector<int> exit_keys)
{
	int key;
	int exit_keys_def[] = {'q', 'Q', 27};
	int mode = 0; // 0->tracking visualization 1-> arrow
	
	cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
	cv::Size winSize(15,15);	//  size of the search window at each pyramid level.
	
	// Default exit keys
	if(exit_keys.empty())	exit_keys = vector<int>(exit_keys_def, exit_keys_def + sizeof(exit_keys_def) / sizeof(exit_keys_def[0]));;

	// Create some random colors
    vector<cv::Scalar> colors;
    cv::RNG rng;
    for(int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(cv::Scalar(r,g,b));
    }

	cv::Mat prev_frame, prev_frame_gray;
	cv::Mat frame, frame_gray;
    vector<cv::Point2f> p0, p1;
    cv::Mat display;

	string winLabel = "(W1) Camera 1 - Optical Flow Lucas-Kanade - GFTT [Bouguet00]";
	cv::namedWindow(WINDOW_CAMERA1, cv::WINDOW_AUTOSIZE);
	cv::setWindowTitle(WINDOW_CAMERA1, winLabel.c_str());
	
    // Take first frame and find corners in it
    camera >> prev_frame;

	if (prev_frame.empty())
	{
			cout << "-- unable to read image from Camera/Video File, sorry.\n";
			return -1;
	}

    cvtColor(prev_frame, prev_frame_gray, cv::COLOR_BGR2GRAY);
    cv::goodFeaturesToTrack(prev_frame_gray, p0, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);   

	// Create a mask image for drawing purposes
    cv::Mat mask = cv::Mat::zeros(prev_frame_gray.size(), prev_frame.type());


	// while there are images ..
	int fn=1;
	while (camera.read(frame))
    {
		fn++;	// frame number

		if (frame.empty())
           return -1;

		cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

        // calculate optical flow
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, p0, p1, status, err, winSize, 2, criteria);

        vector<cv::Point2f> good_new;
        for(unsigned int i = 0; i < p0.size(); i++)
        {
            // Select good points
            if(status[i] == 1) {
                good_new.push_back(p1[i]);
                // draw the tracks
                cv::line(mask,p1[i], p0[i], colors[i], 1);
                if(mode == 0) cv::circle(frame, p1[i], 4, colors[i], 2);
            }
        }
        
		if(mode == 1)
		{
			frame.copyTo(display);
			ShowOpticalFlowArrow(p0, p1, status, display);      
		}		
		else
			cv::add(frame, mask, display);

        cv::imshow(WINDOW_CAMERA1, display);	// show image in a window
		string frameCountLabel = "[Frame " + std::to_string(fn) +"] - " + winLabel; 	// add frame count to window label
		cv::setWindowTitle(WINDOW_CAMERA1, frameCountLabel.c_str());

		 // Now update the previous frame and previous points
        prev_frame_gray = frame_gray.clone();
        p0 = good_new;

        // wait 10ms/DELAY for a keystroke to exit (image window must be on focus) 
        if(IMAGE_FILE) 
			key = cv::waitKey (DELAY);
		else
			key = cv::waitKey (30);

		// Checks if an exit key is pressed
		for(unsigned int i=0; i<exit_keys.size() ; i++)
			if (key == exit_keys[i]) return key;
		
		// Internal options	
		if (key == 'r' || key == 'R' )
		{
			// Reset Optical Flow
			mask.setTo(cv::Scalar(0,0,0));
			cv::goodFeaturesToTrack(prev_frame_gray, p0, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
		}
		else if (key == 'a' || key == 'A' )			mode = 1; // display optical flow as an arrow
		else if (key == 't' || key == 't' )			mode = 0; // display optical flow as tracking features

    }

	return 0;

}


//----------------------------------------------------------------------
// computes Optical Flow  Gunnar Farneback's algorithm. [Farneback2003]
//----------------------------------------------------------------------
// camera: Camera/video file to process
// exit_keys: exit keys (optional)
//----------------------------------------------------------------------
int FarnebackOF(cv::VideoCapture &camera, vector<int> exit_keys)
{
	int key;
	int exit_keys_def[] = {'q', 'Q', 27};
	int mode = 1; // 0->hue visualization 1-> arrow

	cv::Mat prev_frame, prev_frame_gray;
	cv::Mat frame, frame_gray;
	cv::Mat display;

	string winLabel = "(W1) Camera 1 - Dense Optical Flow [Farneback03]";
	cv::namedWindow(WINDOW_CAMERA1, cv::WINDOW_AUTOSIZE);
	cv::setWindowTitle(WINDOW_CAMERA1, winLabel.c_str());
	
	// Default exit keys
	if(exit_keys.empty())	exit_keys = vector<int>(exit_keys_def, exit_keys_def + sizeof(exit_keys_def) / sizeof(exit_keys_def[0]));;
 
 	// Init first frame
    camera >> prev_frame;
	if (prev_frame.empty())
	{
			cout << "-- unable to read image from Camera/Video File, sorry.\n";
			return -1;
	}
    cvtColor(prev_frame, prev_frame_gray, cv::COLOR_BGR2GRAY);

	// Main procesing loop
	int fn=1;
	while (camera.read(frame))
    {
		fn++;	// frame number

        if (frame.empty())
           return -1;
        
		cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        
		cv::Mat flow(prev_frame_gray.size(), CV_32FC2);
        cv::calcOpticalFlowFarneback(prev_frame_gray, frame_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        
		// visualization
		if(mode==1)
		{
			frame.copyTo(display);
			ShowOpticalFlowArrow(flow, display);      
		}
		else
			ShowOpticalFlowHue(flow, display);

		cv::imshow(WINDOW_CAMERA1, display);
		string frameCountLabel = "[Frame " + std::to_string(fn) +"] - " + winLabel; 	// add frame count to window label
		cv::setWindowTitle(WINDOW_CAMERA1, frameCountLabel.c_str());

		// Now update the previous frame 
		frame_gray.copyTo(prev_frame_gray);
		
        // wait 10ms/DELAY for a keystroke to exit (image window must be on focus) 
        if(IMAGE_FILE) 
			key = cv::waitKey (DELAY);
		else
			key = cv::waitKey (10);
    
		// Checks if an exit key is pressed
		for(unsigned int i=0; i<exit_keys.size() ; i++)
			if (key == exit_keys[i]) return key;

		// Internal options	
		if (key == 'a' || key == 'A' )			mode = 1; // display optical flow as an arrow
		else if (key == 'c' || key == 'C' )			mode = 0; // display optical flow as a hue color 

	}

	return 0;

}




//----------------------------------------------------------------------
// computes Optical Flow  DualTV-L1 algorithm. [Zach2007] and [Javier2012]
//----------------------------------------------------------------------
// camera: Camera/video file to process
// exit_keys: exit keys (optional)
//----------------------------------------------------------------------
int DualTVL1OF(cv::VideoCapture &camera,  vector<int> exit_keys)
{
	int key;
	int exit_keys_def[] = {'q', 'Q', 27};
	int mode = 1; // 0->hue visualization 1-> arrow

	cv::Mat prev_frame, prev_frame_gray;
	cv::Mat frame, frame_gray;
 	cv::Mat display;

	string winLabel = "(W1) Camera 1 - Dense Optical Flow DualTV-L1 [Zach07]";
	cv::namedWindow(WINDOW_CAMERA1, cv::WINDOW_AUTOSIZE);
	cv::setWindowTitle(WINDOW_CAMERA1, winLabel.c_str());

	// Default exit keys
	if(exit_keys.empty())	exit_keys = vector<int>(exit_keys_def, exit_keys_def + sizeof(exit_keys_def) / sizeof(exit_keys_def[0]));;
	
	// Init Optical flow algorith
	cv::Ptr<cv::DenseOpticalFlow>  pt_dualTVL1_OF = cv::createOptFlow_DualTVL1();

	// Init first frame
    camera >> prev_frame;
	if (prev_frame.empty())
	{
			cout << "-- unable to read image from Camera/Video File, sorry.\n";
			return -1;
	}	
	cvtColor(prev_frame, prev_frame_gray, cv::COLOR_BGR2GRAY);
    	
	// Main procesing loop	
	int fn=1;
	while (camera.read(frame))
    {
		fn++;	// frame number

        if (frame.empty())
           return -1;
        
		cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
     
		cv::Mat flow(prev_frame_gray.size(), CV_32FC2);
		pt_dualTVL1_OF->calc(prev_frame_gray, frame_gray, flow);
        
		// visualization
		if(mode==1)
		{
			frame.copyTo(display);
			ShowOpticalFlowArrow(flow, display);      
		}
		else
			ShowOpticalFlowHue(flow, display);

		cv::imshow(WINDOW_CAMERA1, display);
		string frameCountLabel = "[Frame " + std::to_string(fn) +"] - " + winLabel; 	// add frame count to window label
		cv::setWindowTitle(WINDOW_CAMERA1, frameCountLabel.c_str());

		// Now update the previous frame 
		frame_gray.copyTo(prev_frame_gray);
		
        // wait 10ms/DELAY for a keystroke to exit (image window must be on focus) 
        if(IMAGE_FILE) 
			key = cv::waitKey (DELAY);
		else
			key = cv::waitKey (30);

		// Checks if an exit key is pressed
		for(unsigned int i=0; i<exit_keys.size() ; i++)
			if (key == exit_keys[i]) return key;

		// Internal options	
		if (key == 'a' || key == 'A' )			mode = 1; // display optical flow as an arrow
		else if (key == 'c' || key == 'C' )			mode = 0; // display optical flow as a hue color 
	       
    }

	pt_dualTVL1_OF->collectGarbage();

	return 0;
}


//----------------------------------------------------------------------
// computes CAMSHIFT object tracking algorithm [Bradski98]  
//	Continuous Adaptative Mean Shift over backprojection image histogram
//----------------------------------------------------------------------
// camera: Camera/video file to process
// exit_keys: exit keys (optional)
//----------------------------------------------------------------------
int CAMShiftOF(cv::VideoCapture &camera,  vector<int> exit_keys)
{
	int key;
	int exit_keys_def[] = {'q', 'Q', 27};
	int mode = 0; // 0->CAMShift 1-> Mean Shift
	int histMode = 0; // 0->hue histogram   1-> intensity hitogram
	int winMode = 1; // 0->mouse click and drag   1-> center point selection (fixed size)
	int winFixSize = 61;	// fixed size for winMode = 1

	cv::Mat roi_hist;
	cv::Mat frame, hsv, backProj;
	cv::Mat display;
	cv::Mat display_hist;

	// Setup the termination criteria, either 10 iteration or move by atleast 1 pt
	cv::TermCriteria term_crit(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1);
    cv::Scalar colorSelection = cv::Scalar(0, 255, 255);

	// Default exit keys
	if(exit_keys.empty())	exit_keys = vector<int>(exit_keys_def, exit_keys_def + sizeof(exit_keys_def) / sizeof(exit_keys_def[0]));;
	
	// Init Optical flow algorithm
	cv::namedWindow(WINDOW_CAMERA1, cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback( WINDOW_CAMERA1, onMouse, 0);

	string winLabel0 = "(W1) Camera 1 - Optical Flow CAMShift [Bradski98]";
	string winLabel1 = "(W1) Camera 1 - Optical Flow Mean Shift";
	if (mode == 1) cv::setWindowTitle(WINDOW_CAMERA1, winLabel1.c_str());
	else cv::setWindowTitle(WINDOW_CAMERA1, winLabel0.c_str());

	// Main procesing loop
	int fn=0;
	while (camera.read(frame))
    {
		fn++;	// frame number

        if (frame.empty())
           return -1;   
		   
		// visualization
		
		frame.copyTo(display);

		if( !SELECTED_OBJECT && SELECTION.x>0 && SELECTION.y>0 && ( (SELECTION.width>0 && SELECTION.height>0) || winMode==1) )
		{
			if(winMode==1) {	
				SELECTION.x -= winFixSize/2; SELECTION.y -= winFixSize/2;	// center
				// Checks that  the window doesn't goes outside the image
				if(SELECTION.x<0) SELECTION.x=0; 
				if(SELECTION.y<0) SELECTION.y=0;
				if(SELECTION.x>=frame.cols-winFixSize) SELECTION.x=frame.cols-winFixSize-1;		
				if(SELECTION.y>=frame.rows-winFixSize) SELECTION.y=frame.rows-winFixSize-1;

				SELECTION.width = winFixSize; SELECTION.height =winFixSize;	// fixed size
			}

			// Calculate histogram for the selected window (Hue or Intensity)
			int res = CalcHistWindowCAMShift(frame, roi_hist, display_hist, histMode);	
			cv::imshow(WINDOW_HISTOGRAM1, display_hist);
			if (histMode == 1) cv::setWindowTitle(WINDOW_PROCESSING1, "(W2) Histogram (Intensity)");
			else cv::setWindowTitle(WINDOW_PROCESSING1, "(W2) Histogram (Hue)");
			SELECTED_OBJECT = true;
		}
	
		if(SELECTED_OBJECT)
		{
			// Meanshift Tracking
			cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);	

			cv::Rect track_window = SELECTION;
			float range_[] = {0, 180};
			const float* range[] = {range_};
			int channels[] = {0};	// Hue
			if(histMode==1)	{ channels[0] = 2; 	range_[1] = 255; }	// Intensity 
		
			// Histogram back projection
			calcBackProject(&hsv, 1, channels, roi_hist, backProj, range);	

			if (mode == 1) {
				// apply camshift to get the new location
				cv::meanShift(backProj, track_window, term_crit);

				cv::cvtColor(backProj, backProj, cv::COLOR_GRAY2BGR);	// converts backprojection to color to draw over it

				// draw original selection
				cv::rectangle(display, SELECTION, colorSelection, 2);
				cv::rectangle(backProj, SELECTION, colorSelection, 2);

				// Draw new location on the image
				cv::rectangle(display, track_window, cv::Scalar(255, 0, 0), 2);
				cv::rectangle(backProj, track_window, cv::Scalar(255, 0, 0), 2);
			}
			else
			{
				// apply camshift to get the new location
				cv::RotatedRect rot_rect = cv::CamShift(backProj, track_window, term_crit);

				cv::cvtColor(backProj, backProj, cv::COLOR_GRAY2BGR);	// converts backprojection to color to draw over it

				// draw original selection
				cv::rectangle(display, SELECTION, colorSelection, 2);
				cv::rectangle(backProj, SELECTION, colorSelection, 2);

				// Draw new location on the image
				cv::ellipse( backProj, rot_rect, cv::Scalar(0,0,255), 3, CV_AA );
				cv::ellipse( display, rot_rect, cv::Scalar(0, 0, 255), 3, CV_AA);
				cv::Point2f rec_points[4];
				rot_rect.points(rec_points);
				for (int i = 0; i < 4; i++)
				{
					cv::line(display, rec_points[i], rec_points[(i + 1) % 4], cv::Scalar(255, 0, 0), 2);
					cv::line(backProj, rec_points[i], rec_points[(i + 1) % 4], cv::Scalar(255, 0, 0), 2);
				}
			}

			cv::imshow(WINDOW_PROCESSING1, backProj);
			if (histMode == 1) cv::setWindowTitle(WINDOW_PROCESSING1, "(W3) CAMShift Histogram BackProjection (Intensity)");
			else cv::setWindowTitle(WINDOW_PROCESSING1, "(W3) CAMShift Histogram BackProjection (Hue)");
		}
    
		cv::imshow(WINDOW_CAMERA1, display);
		string frameCountLabel = "[Frame " + std::to_string(fn) +"] - "; // add frame count to window label
		(mode ==1)? frameCountLabel +=  winLabel1: frameCountLabel +=  winLabel0; 
		cv::setWindowTitle(WINDOW_CAMERA1, frameCountLabel.c_str());
		
		// wait 10ms/DELAY for a keystroke to exit (image window must be on focus) 
        if(IMAGE_FILE) 
			key = cv::waitKey (DELAY);
		else
			key = cv::waitKey (30);

		// Checks if an exit key is pressed
		for(unsigned int i=0; i<exit_keys.size() ; i++)
			if (key == exit_keys[i]) return key;       
		
		// Internal options	
		if (key == 'r' || key == 'R' || 
			key == 'i' || key == 'I' || key == 'h' || key == 'H' ||
			key == 'p' || key == 'P' || key == 'w' || key == 'W' ||
			key == '1' || key == '2' )
		{
			// Reset Search window
			SELECTED_OBJECT = false;
			SELECTION = cv::Rect(0,0,0,0);
			cv::destroyWindow(WINDOW_HISTOGRAM1);
			cv::destroyWindow(WINDOW_PROCESSING1);

			if (key == '1')			mode = 0; // CAM-Shift (Default)
			else if (key == '2')	mode = 1; // Mean-shift
			else if (key == 'i' || key == 'I' )		histMode = 1; // Intensity Hitogram
			else if (key == 'h' || key == 'H' )		histMode = 0; // Hue Histogram (Default)
			else if (key == 'p' || key == 'P' )		winMode = 1; // Select center point (fixed size)
			else if (key == 'w' || key == 'W' )		winMode = 0; // mouse click and drag (Default)
		}
		
    }

	return 0;
}


//----------------------------------------------------------------------
// Calculates and shows histogram in a ROI for CAMShift
//----------------------------------------------------------------------
//----------------------------------------------------------------------
int CalcHistWindowCAMShift(cv::Mat &frame, cv::Mat &roi_hist, cv::Mat &display_hist, int histMode) 
{
	cv::Mat roi, hsv_roi, mask;
	// Calculates ROI histogram
	roi = frame(SELECTION);
	cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);

	if(histMode ==1)
	{	// Intensity Histogram
		// Segments image for all intensity values
		mask = cv::Mat::ones(hsv_roi.size(), CV_8U);
		
		float range_[] = {0, 255};
		const float* range[] = {range_};
		int histSize[] = {256};
		int channels[] = {2};
		cv::calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
		cv::normalize(roi_hist, roi_hist, 0, 255, cv::NORM_MINMAX);

		ShowHistogram(roi_hist, display_hist, false);		
	}
	else
	{	// Hue Histogram
		// Segments image for high intensity and saturation areas
		cv::inRange(hsv_roi, cv::Scalar(0, 30, 20), cv::Scalar(180, 255, 255), mask);

		float range_[] = {0, 180};
		const float* range[] = {range_};
		int histSize[] = {180};
		int channels[] = {0};
		cv::calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
		cv::normalize(roi_hist, roi_hist, 0, 255, cv::NORM_MINMAX);

		ShowHistogram(roi_hist, display_hist, true);
	}
  
	
	if(roi_hist.empty() || cv::sum(roi_hist)==cv::Scalar::all(0))
	{
		cout << ".....Empty Histogram!!!" << endl; 
		return -1;	// empty histogram
	}

	return 0;
}

//----------------------------------------------------------------------
// Mouse callback for CAMShift window selection
//----------------------------------------------------------------------
//----------------------------------------------------------------------
static void onMouse( int event, int x, int y, int, void* ptData )
{

	switch( event )
    {
		case CV_EVENT_LBUTTONDOWN:
			if(!SELECTED_OBJECT)
				SELECTION = cv::Rect(x,y,0,0);
			break;
		case CV_EVENT_LBUTTONUP:
			if( !SELECTED_OBJECT && std::abs(x - SELECTION.x)>15 && std::abs(y - SELECTION.y)>15)
			{
				SELECTION.width = std::abs(x - SELECTION.x);
				SELECTION.height = std::abs(y - SELECTION.y);
			//cout << "Selection-> x:" <<  ptSelection->x << " y:" <<  ptSelection->y << " width:" << ptSelection->width << " height:" << ptSelection->height << endl;
			}
			break;
    }
}




//----------------------------------------------------------------------
// computes Optical Flow for GFTT features with Kalman Filter Algorithm.
//----------------------------------------------------------------------
// camera: Camera/video file to process
// exit_keys: exit keys (optional)
//----------------------------------------------------------------------
int KalmanFilterOF(cv::VideoCapture &camera,  vector<int> exit_keys)
{
	int key;
	int exit_keys_def[] = {'q', 'Q', 27};
	int mode = 0; // 0->tracking visualization 1-> arrow
		
	// Default exit keys
	if(exit_keys.empty())	exit_keys = vector<int>(exit_keys_def, exit_keys_def + sizeof(exit_keys_def) / sizeof(exit_keys_def[0]));;

	// Create some random colors
    vector<cv::Scalar> colors;
    cv::RNG rng;
    for(int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(cv::Scalar(r,g,b));
    }

	cv::Mat prev_frame, prev_frame_gray;
	cv::Mat frame, frame_gray;
    vector<cv::Point2f> p0, p1;
    cv::Mat display;

	cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
	cv::Size winSize(15,15);	//  size of the search window at each pyramid level.

	string winLabel = "(W1) Camera 1 - ptical Flow Kalman Filter - GFTT";
	cv::namedWindow(WINDOW_CAMERA1, cv::WINDOW_AUTOSIZE);
	cv::setWindowTitle(WINDOW_CAMERA1, winLabel.c_str());

	//-------------------------------------------------------------
	// Init Kalman Filter
	// x(k) = A x(k-1) + p_noise(k-1)	state/process equation
	// z(k) = H x(k-1) + m_noise(k)		meassurement equation
	// Q  Process Noise Covariance Matrix 
	// R  Meassurement Noise Covariance Matrix 
	// P' a priori error Covariance Matrix
	// P  a posteriori estimated error Covariance Matrix
	//-------------------------------------------------------------

	// cv::KalmanFilter(int dynamParams, int measureParams, int controlParams=0, int type=CV_32F)
	cv::KalmanFilter KF(4, 2, 0);	
    cv::Mat state(4, 1, CV_32F);	// state x = (px,py,vx,vy) 
    cv::Mat processNoise(4, 1, CV_32F);
    cv::Mat measurement = cv::Mat::zeros(2, 1, CV_32F);	// (px,py)
	
	cv::Mat transitionMatrix(cv::Matx<float,4, 4>(  1, 0, 1, 0, 
												    0, 1, 0, 1, 
												    0, 0, 1, 0, 
													0, 0, 0, 1 ));
	// Meassurement matrix (H)
	cv::Mat measurementMatrix(cv::Matx<float,2, 4>( 1, 0, 0, 0, 
												    0, 1, 0, 0 ));
	// Init Kalman filter data
	KF.transitionMatrix = transitionMatrix;		// Init trasnition matrix (A)
	KF.measurementMatrix = measurementMatrix;	// Meassurement matrix (H)
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));		// Init Process Noise Covariance Matrix  Q
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));	// Init Meassurement Noise Covariance Matrix R
	cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));			// Init a posteriori estimated error Covariance Matrix (P) -> Identity

	// TEST KF with random data
	// cv::randn (InputOutputArray dst, InputArray mean, InputArray stddev)
	cv::randn( state, cv::Scalar::all(0), cv::Scalar::all(0.1) );	// Init state with random values mean 0 and stddev 0.1
	cv::randn( KF.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1) ); // Init a posteriori state estimation with random values mean 0 and stddev 0.1
    
	//-------------------------------------------------------------

	// Create a mask image for drawing purposes
    cv::Mat mask = cv::Mat::zeros(prev_frame_gray.size(), prev_frame.type());

	// Calculate inital state fr Kalman Filter (2 frames)
    // Take first frame and find corners in it
    camera >> prev_frame;
	if (prev_frame.empty())
	{
			cout << "-- unable to read image from Camera/Video File, sorry.\n";
			return -1;
	}
    cvtColor(prev_frame, prev_frame_gray, cv::COLOR_BGR2GRAY);
    cv::goodFeaturesToTrack(prev_frame_gray, p0, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);   

	// Take second frame and find corners in it
    camera >> frame;
	cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

	// calculate optical flow with LK method as initial estimation
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, p0, p1, status, err, winSize, 2, criteria);

    vector<cv::Point2f> good_new;
    for(unsigned int i = 0; i < p0.size(); i++)
    {
        // Select good points
        if(status[i] == 1) {
            good_new.push_back(p1[i]);
            // draw the tracks
            cv::line(mask,p1[i], p0[i], colors[i], 1);
            if(mode == 0) cv::circle(frame, p1[i], 4, colors[i], 2);

			state.at<float>(0) = (float)p1[i].x;
			state.at<float>(1) = (float)p1[i].y;
			state.at<float>(2) = (float)(p1[i].x-p0[i].x);
			state.at<float>(3) = (float)(p1[i].y-p0[i].y);
			KF.statePost = state; // Init a posteriori state estimation
        }
    }
        
	// while there are images ..
	int fn=2;
	while (camera.read(frame))
    {
		fn++;	// frame number

		if (frame.empty())
           return -1;

		cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

        // calculate optical flow
		state = KF.predict();

		// calculate optical flow with LK method as measure
		vector<uchar> status;
		vector<float> err;
		cv::calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, p0, p1, status, err, winSize, 2, criteria);

		vector<cv::Point2f> good_new;
		for(unsigned int i = 0; i < p0.size(); i++)
		{
			// Select good points
			if(status[i] == 1) {
				good_new.push_back(p1[i]);
				// draw the tracks
				cv::line(mask,p1[i], p0[i], colors[i], 1);
				if(mode == 0) cv::circle(frame, p1[i], 4, colors[i], 2);

				measurement.at<float>(0) = (float)p1[i].x;
				measurement.at<float>(1) = (float)p1[i].y;
				state = KF.correct(measurement);		// Updates the predicted state from the measurement.
			}
		}
        
		if(mode == 1)
		{
			frame.copyTo(display);
			ShowOpticalFlowArrow(p0, p1, status, display);      
		}		
		else
			cv::add(frame, mask, display);

        cv::imshow(WINDOW_CAMERA1, display);	// show image in a window
		string frameCountLabel = "[Frame " + std::to_string(fn) +"] - " + winLabel; 	// add frame count to window label
		cv::setWindowTitle(WINDOW_CAMERA1, frameCountLabel.c_str());

		 // Now update the previous frame and previous points
        prev_frame_gray = frame_gray.clone();
        p0 = good_new;

        // wait 10ms/DELAY for a keystroke to exit (image window must be on focus) 
        if(IMAGE_FILE) 
			key = cv::waitKey (DELAY);
		else
			key = cv::waitKey (30);

		// Checks if an exit key is pressed
		for(unsigned int i=0; i<exit_keys.size() ; i++)
			if (key == exit_keys[i]) return key;
		
		// Internal options	
		if (key == 'r' || key == 'R' )
		{
			// Reset Optical Flow
			mask.setTo(cv::Scalar(0,0,0));
			cv::goodFeaturesToTrack(prev_frame_gray, p0, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);

			KF.init(4, 2, 0); // Re-initializes Kalman filter. The previous content is destroyed

			KF.transitionMatrix = transitionMatrix;		// Init trasnition matrix (A)
			KF.measurementMatrix = measurementMatrix;	// Meassurement matrix (H)
			cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));		// Init Process Noise Covariance Matrix  Q
			cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));	// Init Meassurement Noise Covariance Matrix R
			cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));			// Init a posteriori estimated error Covariance Matrix (P) -> Identity

		}
		else if (key == 'a' || key == 'A' )			mode = 1; // display optical flow as an arrow
		else if (key == 't' || key == 't' )			mode = 0; // display optical flow as tracking features

    }

	return 0;
}

//----------------------------------------------------------------------
// computes Point2d from point center displaced in polar coordinates (R,angle)
//----------------------------------------------------------------------
static inline cv::Point calcPoint(cv::Point2f center, double R, double angle)
{
    return center + cv::Point2f((float)cos(angle), (float)-sin(angle))*(float)R;
}

//----------------------------------------------------------------------
// computes Gaussian Mixture-based Background/Foreground Segmentation Algorithm.
// KNN:  [Zivkovic2006] Zoran Zivkovic, Ferdinand van der Heijden, "Efficient adaptive density estimation per image pixel for the task of background subtraction"
// MOG2: [Zivkovic2004] Zoran Zivkovic, "Improved adaptive Gausian mixture model for background subtraction"
//----------------------------------------------------------------------
// camera: Camera/video file to process
// exit_keys: exit keys (optional)
//----------------------------------------------------------------------
int BackgroundSuppressionMOG(cv::VideoCapture &camera,  vector<int> exit_keys)
{
	int key;
	int exit_keys_def[] = {'q', 'Q', 27};
	int mode = 0; // 0->type KNN;   1-> type MOG2; 
	int colorMode = 0;  // 0->Color;   1-> type Gray level;  

	cv::Mat frame, frame_gray, fgmask, bgImage;
 	cv::Mat display;

	// Default exit keys
	if(exit_keys.empty())	exit_keys = vector<int>(exit_keys_def, exit_keys_def + sizeof(exit_keys_def) / sizeof(exit_keys_def[0]));;
	
	cv::namedWindow(WINDOW_CAMERA1, cv::WINDOW_AUTOSIZE);

	string winLabel0 = "(W1) Camera 1 - Background Substraction KNN [Zivkovic2006]";
	string winLabel1 = "(W1) Camera 1 - Background Substraction MOG2 [Zivkovic2004]";
	if (mode == 1) cv::setWindowTitle(WINDOW_CAMERA1, winLabel1.c_str());
	else cv::setWindowTitle(WINDOW_CAMERA1, winLabel0.c_str());
	
	//create Background Subtractor objects
	cv::Ptr<cv::BackgroundSubtractor> pBackSubMOG2 = cv::createBackgroundSubtractorMOG2();
	cv::Ptr<cv::BackgroundSubtractor> pBackSubKNN = cv::createBackgroundSubtractorKNN();
   	
	// Main procesing loop	
	int fn=0;
	while (camera.read(frame))
    {
		fn++;	// frame number

        if (frame.empty())
           return -1;
        
		cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
     
		frame.copyTo(display);

		if (mode == 1)	
		{// MOG2
			if (colorMode == 1)
				pBackSubMOG2->apply(frame_gray, fgmask);
			else
				pBackSubMOG2->apply(frame, fgmask);

			pBackSubMOG2->getBackgroundImage(bgImage);
			cv::add(frame, cv::Scalar(0, 100, 100), display, fgmask);
		}
		else  
		{ // KNN
			if (colorMode == 1)
				pBackSubKNN->apply(frame_gray, fgmask);
			else
				pBackSubKNN->apply(frame, fgmask);

			pBackSubKNN->getBackgroundImage(bgImage);
			cv::add(frame, cv::Scalar(0, 100, 100), display, fgmask);
		}

		// Show images
		cv::imshow(WINDOW_CAMERA1, display);
		string frameCountLabel = "[Frame " + std::to_string(fn) +"] - "; // add frame count to window label
		(mode ==1)? frameCountLabel +=  winLabel1: frameCountLabel +=  winLabel0; 
		cv::setWindowTitle(WINDOW_CAMERA1, frameCountLabel.c_str());

		cv::imshow(WINDOW_PROCESSING1, fgmask);	// Foreground Segmentation
		cv::setWindowTitle(WINDOW_PROCESSING1, "(W2) Background Substraction (Foreground Mask)");
		
		if (!bgImage.empty())
		{
			cv::imshow(WINDOW_PROCESSING2, bgImage);	// Foreground Segmentation
			cv::setWindowTitle(WINDOW_PROCESSING2, "(W3) Background Image");
		}

        // wait 10ms/DELAY for a keystroke to exit (image window must be on focus) 
        if(IMAGE_FILE) 
			key = cv::waitKey (DELAY);
		else
			key = cv::waitKey (30);

		// Checks if an exit key is pressed
		for(unsigned int i=0; i<exit_keys.size() ; i++)
			if (key == exit_keys[i]) return key;

		// Internal options	
		if (key == 'r' || key == 'R' || 
			key == 'i' || key == 'I' || key == 'c' || key == 'C' ||
			key == '1' || key == '2')
		{
			// Reset background model

			if (key == '1')					mode = 0; // KNN (Default)
			else if (key == '2')			mode = 1; // MOG2
			else if (key == 'i' || key == 'I' )		colorMode = 1; // Intensity (gray level)
			else if (key == 'c' || key == 'C' )		colorMode = 0; // Color (Default)

			cv::destroyWindow(WINDOW_PROCESSING1);
			cv::destroyWindow(WINDOW_PROCESSING2);
		}
    }

	return 0;
}


