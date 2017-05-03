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
#include </home/pi/MAVLink/ardupilotmega/mavlink.h>

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
static const int DEFAULT_PORT = 14552;	// ArduPilot port
static const int BUFFER_SIZE = 2048;
static const int DEADZONE_LEFT_BOUND = 288;
static const int DEADZONE_RIGHT_BOUND = 352;
static const int DEADZONE_NEAR_BOUND = 40; //Totally made up at the moment
static const int DEADZONE_FAR_BOUND = 10; //Totally made up 
static const int DEG_LONGITUDE_ONE_METER = 90; //Degrees * 10^-7
static const int DEG_LATITUDE_ONE_METER = 133; //Degrees * 10^-7
static const int METERS_PER_WAYPOINT = 10;


const uint8_t CC_SYSID = 32;
const uint8_t CC_COMPID = 1;
const uint8_t TARGET_ID = 1;	
const uint32_t DEFAULT_TIMESTAMP = 0;
const int32_t DEFAULT_LATITUDE = 0;
const int32_t DEFAULT_LONGITUDE = 0;
const float DEFAULT_ALTITUDE = 0;
const float DEFAULT_HEADING = 0;
const uint16_t ENABLE_POSITION_BITS = 0b0000111111111000;
const uint8_t NO_MODE = 0b00000000;

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
	{}
	
	//constructor to allow for 
	LocSizeSide(Rect rect, char Side)
	{
		x = rect.x + (rect.width / 2); //x and y are set to middle
		y = rect.y + (rect.height / 2);
		w = rect.width;
		h = rect.height;
		side = Side;
	}

	int x, y, w, h; //x and y are middle of pattern, not top left
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

void classifier(Mat& img, vector<LocSizeSide>& detections, CascadeClassifier cascade, char side, int& filterStage, LocSizeSide& previousDetection, int predictVector[4])
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

void classifierManager(RingBuffer& buffer, DetectionBuffer& detectionBeingFollowed, int& detectionStage)
{
	Mat img;
	
	CascadeClassifier backCascade;
	CascadeClassifier frontCascade;
	//CascadeClassifier sideCascade; doesn't exist yet
	
	vector<LocSizeSide> detections;

	LocSizeSide previousDetection;
	int predictVector[4] = {};
	int closestPatternIndex = 0;

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

			detectionBeingFollowed = detections[j];

			predictVector[0] = detections[j].x - previousDetection.x;
			predictVector[1] = detections[j].y - previousDetection.y;
			predictVector[2] = detections[j].w - previousDetection.w;
			predictVector[3] = detections[j].h - previousDetection.h;
			
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

void udp_server(DetectionBuffer& detection, int& detectionStage)
{
	int sock;					// socket id
	unsigned short servPort;			// port number
	struct sockaddr_in servAddr;			// our server address struct
	struct sockaddr_in clientAddr;			// remote client address struct
	socklen_t addrLen = sizeof(clientAddr);		// length of addresses
	int recvlen;					// bytes received
	unsigned char buffer[BUFFER_SIZE];		// receive buffer
	
	mavlink_message_t msg;
	mavlink_status_t status;		
	uint8_t parser;	
	
	uint16_t send_length;
	
	uint8_t flight_mode = 0b00000000;
	
	uint32_t timestamp = DEFAULT_TIMESTAMP;
	uint32_t current_lat = DEFAULT_LATITUDE;
	uint32_t current_lon = DEFAULT_LONGITUDE;
	float current_alt = DEFAULT_ALTITUDE;	
	float current_hdg = DEFAULT_HEADING;		
	
	float heading = -1;

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
	
	while(1){
		// printf("Waiting on port %d\n", servPort);
		recvlen = recvfrom(sock, buffer, MAVLINK_MAX_PACKET_LEN, 0, (struct sockaddr *)&clientAddr, &addrLen);
		// printf("Received %d bytes.\n", recvlen);
		if(recvlen > 0){
			buffer[recvlen] = 0;
			
			for(int i=0; i < recvlen; i++){
				parser = buffer[i];
				
 				if(mavlink_parse_char(MAVLINK_COMM_0, parser, &msg, &status)){
					// cout << "DEBUG msgid: " << msg.msgid << endl;
					switch(msg.msgid){
						case MAVLINK_MSG_ID_HEARTBEAT:
							mavlink_heartbeat_t hb;
							mavlink_msg_heartbeat_decode(&msg, &hb);
							
							flight_mode = hb.base_mode;
							
							#ifdef MSG_DEBUG
								cout << endl;
								cout << ">> Heartbeat Message Received:" << endl;
								cout << "custom_mode: " << (int) hb.custom_mode << endl;
								cout << "Type: " << (int) hb.autopilot << endl;
								cout << "base_mode: " << (int) hb.base_mode << endl;
								cout << "system_status: " << (int) hb.system_status << endl;
								cout << "mavlink_version: " << (int) hb.mavlink_version << endl;
								cout << endl;
							#endif
							break;

						case MAVLINK_MSG_ID_GLOBAL_POSITION_INT:
							mavlink_global_position_int_t gpi_status;
							mavlink_msg_global_position_int_decode(&msg, &gpi_status);
							
							// Noting GPI status and converting current position to floats
							timestamp = gpi_status.time_boot_ms;
							current_lat = gpi_status.lat;
							current_lon = gpi_status.lon;
							current_alt = ((float) gpi_status.relative_alt)/1000;	// millimeters --> meters
							current_hdg = ((float) gpi_status.hdg)/100;				// 1E2 --> degrees
							
							#ifdef MSG_DEBUG
								cout << endl;
								cout << ">> Global Position Message Received:" << endl;
								cout << "Timestamp: " << gpi_status.time_boot_ms << endl;
								cout << "Latitude: " << (float) ((float) gpi_status.lat)/10000000 << endl;
								cout << "Longitude: " << (float) ((float) gpi_status.lon)/10000000 << endl;
								cout << "Altitude: " << gpi_status.alt << endl;
								cout << "Relative Altitude: " << (float) ((float) gpi_status.relative_alt)/1000 << endl;
								cout << "Ground Velocity (X): " << gpi_status.vx << endl;
								cout << "Ground Velocity (Y): " << gpi_status.vy << endl;
								cout << "Ground Velocity (Z): " << gpi_status.vz << endl;
								cout << "Vehicle heading (Yaw): " << current_hdg << endl;
								cout << endl;
							#endif
							break;
							
						// default:
							// cout << endl << ">> Received unsupported message." << endl << endl;
					}
				}
			}
			// printf("\nReceived message: '%s'\n", buffer);
			
			if(timestamp == DEFAULT_TIMESTAMP){
			
				// Request Data Stream Command:
				 uint8_t stream_id = 0;			// See MAV_DATA_STREAM ENUM (supposedly, maybe not supported in ardupilot)
				 uint16_t request_rate = 1;		// Rate to send message per second
				 int32_t interval_microseconds = 100000;
				mavlink_msg_request_data_stream_pack(CC_SYSID, CC_COMPID, &msg, TARGET_ID, MAV_COMP_ID_ALL, stream_id, request_rate, 1);
				
				// Sending the MAVLink message to buffer and then over UDP protocol to client
				 send_length = mavlink_msg_to_send_buffer(buffer, &msg);
				 sendto(sock, buffer, MAVLINK_MAX_PACKET_LEN, 0, (struct sockaddr *)&clientAddr, addrLen);
				// cout << "request sent" << endl;	
			}
			
			// Send MAVLink commands
			//detectLk.lock();
			//detectCV.wait(detectLk, [] {return newDetection;});
			//followPatternReading = true;
			detectionToFollow = detection.getLast();
			//followPatternReading = false;
			//newDetection = false;
			//detectCV.notify_all();
			
			uint8_t confirmation = 1;
			uint32_t desired_lat = current_lat;
			uint32_t desired_lon = current_lon;
			float desired_alt = current_alt;

			if(detectionToFollow.x < DEADZONE_LEFT_BOUND){
				if(detectionToFollow.w < DEADZONE_FAR_BOUND){
					//move left and forward
					//cout << "move left and forward";
					heading = 315;
				}else if(detectionToFollow.w > DEADZONE_NEAR_BOUND){
					//move left and backwards
					//cout << "move left and backwards";
					heading = 225;
					//set heading to 225
				}else{
					//move left
					//cout << "move left";
					heading = 270;
					//set heading to 270
				}
			}else if(detectionToFollow.x > DEADZONE_RIGHT_BOUND){
				if(detectionToFollow.w < DEADZONE_FAR_BOUND){
					//move right and forward
					//cout << "move right and forward";
					//set heading to 45
					heading = 45;
				}else if(detectionToFollow.w > DEADZONE_NEAR_BOUND){
					//move right and backwards
					//cout << "move right and backwards";
					//set heading to 135
					heading = 135;
				}else{
					//move right
					//cout << "move right";
					//set heading to 90
					heading = 90;
				}
			}else{
				if(detectionToFollow.w < DEADZONE_FAR_BOUND){
					//move forward
					//cout << "move forward";
					//set heading to 0
					heading = 0;
				}else if(detectionToFollow.w > DEADZONE_NEAR_BOUND){
					//move backwards
					//cout << "move backwards";
					//set heading to 180
					heading = 180;
				}else{
					//nothing
					cout << "stay still";
					//set heading to -1
					heading = -1;
				}
			}

			if(heading >= 0 && detectionStage != 0){
				//waypoint is current longitude+sin(yaw + heading / 10000) * METERS_PER_WAYPOINT * DEG_LONGITUDE_ONE_METER, current lattitude+cos(yaw + heading / 10000) * METERS_PER_WAYPOINT * DEG_LATITUDE_ONE_METER
				desired_lon = current_lon+sin(current_hdg + heading) * METERS_PER_WAYPOINT * DEG_LONGITUDE_ONE_METER;
				desired_lat = current_lat+cos(current_hdg + heading) * METERS_PER_WAYPOINT * DEG_LONGITUDE_ONE_METER;
				mavlink_msg_set_position_target_global_int_pack(CC_SYSID, CC_COMPID, &msg, timestamp,
					TARGET_ID, MAV_COMP_ID_ALL, MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, ENABLE_POSITION_BITS,
					desired_lat, desired_lon, desired_alt, 0, 0, 0, 0, 0, 0, 0, 0);
				send_length = mavlink_msg_to_send_buffer(buffer, &msg);
				sendto(sock, buffer, MAVLINK_MAX_PACKET_LEN, 0, (struct sockaddr *)&clientAddr, addrLen);
				// cout << "Message sent, desired alt: " << desired_alt << endl;						
			}
			this_thread::sleep_for(milliseconds(10));
		}
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
	
	int detectionStage = 0;
	
	// Initialize threads
	running = true;
	thread cam(cameraFeed, ref(camera), ref(imgBuffer));
	thread classMngr(classifierManager, ref(imgBuffer), ref(detectionToFollow), ref(detectionStage));
	thread mavlinkServer(udp_server, ref(detectionToFollow), ref(detectionStage));

	// Exit camera feed and classifier manager
	cam.join();
	classMngr.join();
	mavlinkServer.join();

	return 0;
}
