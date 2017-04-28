//Current working version
//Runs at 4 fps at 640, 480
//Uses ~65% of RPi processing power

//TO DO:
//	Test initializing and joining classifier threads as opposed to pooling them
//	Take info in detection vector to do flight commands
//	Clean detection vector after interpreter for instructions acquires needed data

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <raspicam/raspicam_cv.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h> 	// size_t, ssize_t
#include <sys/socket.h> // socket funcs
#include <netinet/in.h> // sockaddr_in
#include <arpa/inet.h> 	// htons, inet_pton
#include <unistd.h> 	// close

using namespace std;
using namespace cv;
using namespace raspicam;
using namespace chrono;

//buffer and classifier sync primitives
mutex bufMtx;
mutex classMtx;
mutex detectionMtx;

condition_variable bufCV;
condition_variable classCV;
condition_variable detectCV;

atomic_uchar classifiersFinished;

bool buffersReady = false;
bool startClassifiers = false;
bool running = false;
bool classificationDone = false;
bool followPatternReading = false;
bool followPatternWriting = false;
bool newDetection = false;

static const char BACK = 'b';
static const char FRONT = 'f';
static const char SIDE = 's';
static const char NUM_PATTERNS = 2;
static const int FRAMES = 1000;
static const int DEFAULT_PORT = 14550;	// ArduPilot port
static const int BUFFER_SIZE = 2048;
static const int DEADZONE_LEFT_BOUND = 288;
static const int DEADZONE_RIGHT_BOUND = 352;
static const int DEADZONE_NEAR_BOUND = 40; //Totally made up at the moment
static const int DEADZONE_FAR_BOUND = 10; //Totally made up 
static const int DEG_LONGITUDE_ONE_METER = 90; //Degrees * 10^-7
static const int DEG_LATITUDE_ONE_METER = 133; //Degrees * 10^-7
static const int METERS_PER_WAYPOINT = 2;

static const string backCascadeName = "/home/pi/seniordesign/classifiers/backCascade/cascade.xml";
static const string frontCascadeName = "/home/pi/seniordesign/classifiers/frontCascade/cascade.xml";
//static const string sideCascadeName = "/home/pi/seniordesign/classifiers/side/cascade.xml";

static const int WIDTH = 640;
static const int HEIGHT = 480;

// Creates triple buffer to store camera images
// Allows for classifier manager to grab newest image without contesting
//	for resources with camera feed
// Added 4th buffer due to contestion for some resource (unclear if this
//	buffer was being contested but issue hasn't arrised since switch)
class RingBuffer{
public:	
	// Constructor for ring buffer, creates 4 buffers in bi-directional circular link
	RingBuffer(){
		currentNode = new Node; //create buff1
		currentNode->next = new Node; //create buff2 as buff1 next
		currentNode->next->prev = currentNode; //link buff2 prev to buff1
		currentNode->next->next = new Node; //create buff3 as next for 2
		currentNode->next->next->prev = currentNode->next; //link 3 prev to 2
		currentNode->prev = new Node; //create buff4 as buff1 prev
		currentNode->prev->next = currentNode; //link buff4 next to buff1
		currentNode->prev->prev = currentNode->next->next; //link buff4 prev to buff3
		currentNode->next->next->next = currentNode->prev; //link buff3 next to buff4
	}
	
	// overloaded assignement operator to add images from camera feed easily
	void operator=(const Mat& image){
		currentNode->img = image;
		currentNode = currentNode->next;
	}
	
	// destructor to clean up buffers
	~RingBuffer(){
		delete currentNode->next->next;
		delete currentNode->next;
		delete currentNode->prev;
		delete currentNode;
	}
	
	//Access last saved image
	Mat getLast(){
		return currentNode->prev->img;
	}
	
private:
	//Nodes for buffer (each one is a buffer with links to the other 2)
	struct Node{
	public:
		Node(){
			img = Mat::zeros(WIDTH, HEIGHT, CV_8UC1);
			next = nullptr;
			prev = nullptr;
		}
		
		Mat img;
		Node* next;
		Node* prev;
	};
	
	//base node
	Node * currentNode = nullptr;
};

struct LocSizeSide
{
public:
	//base constructor does nothing
	LocSizeSide()
	{
		x = WIDTH/2;
		y = HEIGHT/2;
		w = 0;
		h = 0;
		side = FRONT;
		score = 0.0;
	}
	
	//constructor to allow for 
	LocSizeSide(Rect rect, char Side, float Score)
	{
		x = rect.x + (rect.width / 2); //x and y are set to middle
		y = rect.y + (rect.height / 2);
		w = rect.width;
		h = rect.height;
		side = Side;
		score = Score;
	}

	int x, y, w, h; //x and y are middle of pattern, not top left
	float score; 
	char side;
};

class DetectionBuffer{
public:	
	// Constructor for ring buffer, creates 4 buffers in bi-directional circular link
	DetectionBuffer(){
		currentNode = new Node; //create buff1
		currentNode->next = new Node; //create buff2 as buff1 next
		currentNode->next->prev = currentNode; //link buff2 prev to buff1
		currentNode->next->next = new Node; //create buff3 as next for 2
		currentNode->next->next->prev = currentNode->next; //link 3 prev to 2
		currentNode->prev = new Node; //create buff4 as buff1 prev
		currentNode->prev->next = currentNode; //link buff4 next to buff1
		currentNode->prev->prev = currentNode->next->next; //link buff4 prev to buff3
		currentNode->next->next->next = currentNode->prev; //link buff3 next to buff4
	}
	
	// overloaded assignement operator to add images from camera feed easily
	void operator=(const LocSizeSide& detect){
		currentNode->detection = detect;
		currentNode = currentNode->next;
	}
	
	// destructor to clean up buffers
	~DetectionBuffer(){
		delete currentNode->next->next;
		delete currentNode->next;
		delete currentNode->prev;
		delete currentNode;
	}
	
	//Access last saved image
	LocSizeSide getLast(){
		return currentNode->prev->detection;
	}
	
private:
	//Nodes for buffer (each one is a buffer with links to the other 2)
	struct Node{
	public:
		Node(){
			next = nullptr;
			prev = nullptr;
		}
		
		LocSizeSide detection;
		Node* next;
		Node* prev;
	};
	
	//base node
	Node * currentNode = nullptr;
};

void cameraFeed(RaspiCam_Cv& camera, RingBuffer& buffer)
{
	unique_lock<mutex> lk(bufMtx);
	
	Mat img;
	
	//fill buffer
	for(int i = 0; i < 4; i++){
		camera.grab();
		camera.retrieve(img);
		buffer = img;
	}
	
	//buffer full, classifiers can run
	buffersReady = true;
	lk.unlock();
	bufCV.notify_one();
	
	//start image capture process
	while(running){
		camera.grab();
		camera.retrieve(img);
		buffer = img;
	}
}

void classifier(Mat& img, vector<LocSizeSide>& detections, CascadeClassifier cascade, char side, int filterStage&, LocSizeSide& previousDetection, int predictVector[4])
{
	// Create local detection vector
	vector<Rect> localDetections;
	// Create lock and defer
	unique_lock<mutex> lk(classMtx, defer_lock);
	
	float detectionScore;
	int differenceVector[4] = {};

	while(running){
		// Wait for classifier manager
		lk.lock();
		classCV.wait(lk, []{return startClassifiers;});
		
		// Allow other threads to unlock 
		// Matt is there a better way to implement a barrier in C++11?
		lk.unlock(); 
		classCV.notify_all();
		
		// Detect patterns
		cascade.detectMultiScale(img, localDetections, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
		
		// Add patterns to main detection vector including side info via emplacement which moves Rect data into LocSizeSide wrapper
		for(size_t i = 0; i < localDetections.size(); i++){
			switch(filterStage){
				case 1:
					detectionScore = abs(localDetections[j].x - previousDetection.x) + abs(localDetections[j].y - previousDetection.y) + abs(localDetections[j].w - previousDetection.w) + abs(localDetections[j].h - previousDetection.h);
					break;
				case 2:
					differenceVector[0] = abs(localDetections[j].x - previousDetection.x);
					differenceVector[1] = abs(localDetections[j].y - previousDetection.y);
					differenceVector[2] = abs(localDetections[j].w - previousDetection.w);
					differenceVector[3] = abs(localDetections[j].h - previousDetection.h);
					detectionScore = abs(((float)differenceVector[0] - (float)predictVector[0])/(float)predictVector[0]) + abs(((float)differenceVector[1] - (float)predictVector[1])/(float)predictVector[1]) + abs(((float)differenceVector[2] - (float)predictVector[2])/(float)predictVector[2]) + abs(((float)differenceVector[3] - (float)predictVector[3])/(float)predictVector[3]);
					break;
				default:
					detectionScore = 0.0;
			}
			detections.emplace_back(localDetections[i], side, detectionScore);
			if(filterStage == 0)
				break;
		}
		
		// Increment classifiers finished and notify manager, once at 3
		classifiersFinished++;
		startClassifiers = false;
		
		// If last classifier to finish allow classifier manager to prepare next frame
		if(classifiersFinished == NUM_PATTERNS)
			classificationDone = true;
		
		classCV.notify_all();
	}
	
}

void classifierManager(RingBuffer& buffer, DetectionBuffer& detectionBeingFollowed)
{
	Mat img;
	
	CascadeClassifier backCascade;
	CascadeClassifier frontCascade;
	//CascadeClassifier sideCascade; doesn't exist yet
	
	vector<LocSizeSide> detections;

	LocSizeSide previousDetection;
	int predictVector[4] = {};
	int closestPatternIndex = 0;
	int detectionStage = 0; //0: no detections found, 1: one detection found previously, 2: more than one found previously

	if(!backCascade.load(backCascadeName)){
		running = false;
		cout << "Error loading back cascade" << endl;
		return;
	}
	
	if(!frontCascade.load(frontCascadeName)){
		running = false;
		cout << "Error loading front cascade" << endl;
		return;
	}
	/*	THIS DOESN'T EXIST YET
	if(!sideCascade.load(sideCascadeName)){
		running = false;
		cout << "Error loading side cascade" << endl;
		return;
	}*/
	
	// Lock classifiers
	unique_lock<mutex> classLk(classMtx);
	//unique_lock<mutex> detectLk(detectionMtx, defer_lock);

	// Currently just using back cascade on all 3
	thread backClassifier(classifier, ref(img), ref(detections), backCascade, BACK, detectionStage, ref(previousDetection), predictVector);
	thread frontClassifier(classifier, ref(img), ref(detections), frontCascade, FRONT, detectionStage, ref(previousDetection), predictVector);
	//thread sideClassifier(classifier, ref(img), ref(detections), backCascade, SIDE, detectionStage, ref(previousDetection), predictVector);
	
	// Wait for buffer to be filled
	unique_lock<mutex> bufLk(bufMtx);
	bufCV.wait(bufLk, []{return buffersReady;});

	cout << "starting classifiers" << endl;
	
	// Run time analysis
	time_point<system_clock> start, end;
	start = system_clock::now();
	
	// Currently for loop because there are no conditions to stop otherwise
	for(int i = 0; i < FRAMES; i++){
		// Grab image and prepare for classifiers
		img = buffer.getLast();
		equalizeHist(img, img);
		classifiersFinished = 0;
		classificationDone = false;
		
		// Tell classifiers to run
		startClassifiers = true;
		classLk.unlock();
		classCV.notify_all();
		
		// Wait until classifiers are finished running
		classLk.lock();
		classCV.wait(classLk, []{return classificationDone;});
		// ClassLk is reacquired after wait()

		//Parse detections for lowest score
		if(detection.empty()){
			detectionStage = 0;
			cout << "Zero" << endl;
		}else if(detectionStage == 0){
			detectionBeingFollowed = detections[0];
			detectionStage = 1;
			cout << "One" << endl;
		}else{
			closestPatternIndex = 0;

			previousDetection = detectionBeingFollowed.getLast();

			for(int j = 1; i < detections.size(); i++){
				if(detections[j].score < detections[closestPatternIndex].score)
					closestPatternIndex = j;
			}

			detectionBeingFollowed = detections[0];

			predictVector[0] = detectionBeingFollowed.x - previousDetection.x;
			predictVector[1] = detectionBeingFollowed.y - previousDetection.y;
			predictVector[2] = detectionBeingFollowed.w - previousDetection.w;
			predictVector[3] = detectionBeingFollowed.h - previousDetection.h;
			
			detectionStage = 2;
		}

		detections.clear();
	}
	running = false;
	
	img = buffer.getLast();
	equalizeHist(img, img);
	classifiersFinished = 0;
	classificationDone = false;
	
	// Tell classifiers to run
	startClassifiers = true;
	classLk.unlock();
	classCV.notify_all();
	
	// Run time analysis
	end = system_clock::now();
	duration<double> elapsed_seconds = end-start;
	
	cout << "Exiting loop. " << FRAMES << " frames." << endl 
		<< "Total time: " << elapsed_seconds.count() << endl
		<< "Average frame rate: " << ((double) FRAMES / elapsed_seconds.count()) << endl;
	
	// End classifiers and camera feed
	backClassifier.join();
	frontClassifier.join();
	//sideClassifier.join();
	
}

void udp_server(DetectionBuffer& detection)
{
	int sock;					// socket id
	unsigned short servPort;			// port number
	struct sockaddr_in servAddr;			// our server address struct
	struct sockaddr_in clientAddr;			// remote client address struct
	socklen_t addrLen = sizeof(clientAddr);		// length of addresses
	int recvlen;					// bytes received
	unsigned char buffer[BUFFER_SIZE];		// receive buffer
	bool validMessage = false;			// bool to check for if a valid message was received
	char* send_message = "test";
	  
	int heading = -1;

	//unique_lock<mutex> detectLk(detectionMtx, defer_lock);
	LocSizeSide detectionToFollow;

	// 1. Create a UDP socket
	sock = socket(AF_INET, SOCK_DGRAM, 0);
	if (sock < 0) {
		cerr << "Error with create socket" << endl;
		exit (-1);
	}	
	else{
		printf("Socket created with ID %d\n", sock);
	}
	
	// 2. Define server address struct
	servPort = DEFAULT_PORT;
	printf("Using default port number %d.\n", DEFAULT_PORT);	
	
	// Set the fields for servAddr struct
	// INADDR_ANY is a wildcard for any IP address
	//	- binds socket to all available interfaces
	memset((char *)&servAddr, 0, sizeof(servAddr));
	servAddr.sin_family = AF_INET; 					// always AF_INET
	servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
	servAddr.sin_port = htons(servPort);
	
	// 3. Bind socket to server address
	if(bind(sock, (struct sockaddr *) &servAddr, sizeof(servAddr)) < 0){
		cerr << "Error with bind socket";
		exit(-1);
	}	
	
	// Main loop to receive/send data with clients
	while(!validMessage){
		printf("Waiting on port %d\n", servPort);
		recvlen = recvfrom(sock, buffer, BUFFER_SIZE, 0, (struct sockaddr *)&clientAddr, &addrLen);
		printf("Received %d bytes.\n", recvlen);
		if(recvlen > 0){
			buffer[recvlen] = 0;
			printf("Received message: '%s'\n", buffer);
			validMessage = true;
		}
	}
	
	while(1){
		// Send MAVLink commands
		//detectLk.lock();
		//detectCV.wait(detectLk, [] {return newDetection;});
		//followPatternReading = true;
		detectionToFollow = detection.getLast();
		//followPatternReading = false;
		//newDetection = false;
		//detectCV.notify_all();

		if(detectionToFollow.x < DEADZONE_LEFT_BOUND){
			if(detectionToFollow.w < DEADZONE_FAR_BOUND){
				//move left and forward
				cout << "move left and forward";
				heading = 31500;
			}else if(detectionToFollow.w > DEADZONE_NEAR_BOUND){
				//move left and backwards
				cout << "move left and backwards";
				heading = 22500;
				//set heading to 225
			}else{
				//move left
				cout << "move left";
				heading = 27000;
				//set heading to 270
			}
		}else if(detectionToFollow.x > DEADZONE_RIGHT_BOUND){
			if(detectionToFollow.w < DEADZONE_FAR_BOUND){
				//move right and forward
				cout << "move right and forward";
				//set heading to 45
				heading = 4500;
			}else if(detectionToFollow.w > DEADZONE_NEAR_BOUND){
				//move right and backwards
				cout << "move right and backwards";
				//set heading to 135
				heading = 13000;
			}else{
				//move right
				cout << "move right";
				//set heading to 90
				heading = 9000;
			}
		}else{
			if(detectionToFollow.w < DEADZONE_FAR_BOUND){
				//move forward
				cout << "move forward";
				//set heading to 0
				heading = 0;
			}else if(detectionToFollow.w > DEADZONE_NEAR_BOUND){
				//move backwards
				cout << "move backwards";
				//set heading to 180
				heading = 18000;
			}else{
				//nothing
				cout << "stay still";
				//set heading to -1
				heading = -1;
			}
		}

		switch(heading){
			case -1:
				//waypoint is where drone is
				break;

			default:
				//waypoint is current longitude-sin(yaw + heading / 1000) * METERS_PER_WAYPOINT * DEG_LONGITUDE_ONE_METER, current lattitude+cos(yaw + heading / 1000) * METERS_PER_WAYPOINT * DEG_LATITUDE_ONE_METER
		}
		this_thread::sleep_for(milliseconds(10));
		// Send function
		sendto(sock, send_message, strlen(send_message), 0, (struct sockaddr *)&clientAddr, addrLen);
	}

	 close(sock);	
}

int main()
{
	// Create buffer
	RingBuffer imgBuffer;
	DetectionBuffer detectionToFollow;
	//LocSizeSide detectionToFollow;

	// Create and set up camera
	RaspiCam_Cv camera;
	camera.set(CV_CAP_PROP_FORMAT, CV_8UC1); // Set camera to 8-bit grayscale
	camera.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH); // Set width
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT); // Set height
	camera.open();
	
	// Initialize threads
	running = true;
	thread cam(cameraFeed, ref(camera), ref(imgBuffer));
	thread classMngr(classifierManager, ref(imgBuffer), ref(detectionToFollow));
	thread mavlinkServer(udp_server, ref(detectionToFollow));

	// Exit camera feed and classifier manager
	cam.join();
	classMngr.join();
	mavlinkServer.join();

	return 0;
}
