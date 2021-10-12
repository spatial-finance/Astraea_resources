#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    ofSetWindowShape(640, 480);
    
    normalize = true;
    
    // default settings
    oscDestination = OSC_DESTINATION_DEFAULT;
    oscAddress = OSC_ADDRESS_ROOT_DEFAULT;
    oscPort = OSC_PORT_DEFAULT;

    // load settings from file
    ofxXmlSettings xml;
    xml.load("settings_facetracker2.xml");
//    xml.setTo("FaceTracker2OSC");
//    oscDestination = xml.getValue("ip");
//    oscPort = ofToInt(xml.getValue("port"));
//    oscAddress = xml.getValue("address");

    
    oscDestination = xml.getValue("FaceTracker2OSC:ip", OSC_DESTINATION_DEFAULT);
    oscPort = xml.getValue("FaceTracker2OSC:port", OSC_PORT_DEFAULT);
    oscAddress = xml.getValue("FaceTracker2OSC:address", OSC_ADDRESS_ROOT_DEFAULT);

    
    // setup tracker and cam
    grabber.setup(640, 480);
    tracker.setup(ofToDataPath("../../../../data/shape_predictor_68_face_landmarks.dat"));

    // addresses + setup osc
    osc.setup(oscDestination, oscPort);
    
    // setup gui
    gui.setup();
    gui.setPosition(12, 60);
    gui.setName("FaceTracker2OSC");
    gui.add(normalize.set("normalize", true));
}

//--------------------------------------------------------------
void ofApp::update(){
    grabber.update();
    if(!grabber.isFrameNew()){
        return;
    }
    
    tracker.update(grabber);

    vector<ofxFaceTracker2Instance> instances = tracker.getInstances();
    if (instances.size() == 0) {
        return;
    }

    ofxFaceTracker2Landmarks landmarks = instances[0].getLandmarks();
    vector<glm::vec2> points = landmarks.getImagePoints();
    ofRectangle bb = instances[0].getBoundingBox();
    
    ofxOscMessage msg;
    msg.setAddress(oscAddress);
    for (int i=0; i<points.size(); i++) {
        if (normalize) {
            msg.addFloatArg((points[i].x-bb.getMinX()) / bb.getWidth());
            msg.addFloatArg((points[i].y-bb.getMinY()) / bb.getHeight());
        }
        else {
            msg.addFloatArg(points[i].x);
            msg.addFloatArg(points[i].y);
        }
    }

    osc.sendMessage(msg);
    oscMessageString = "Sending OSC to "+oscDestination+", port "+ofToString(oscPort)+": address "+oscAddress+" -> "+ofToString(msg.getNumArgs())+" values";
}

//--------------------------------------------------------------
void ofApp::draw(){
    grabber.draw(0, 0);
    tracker.drawDebug();
    tracker.drawDebugPose();
    
    // Draw text UI
    ofDrawBitmapStringHighlight("Framerate : "+ofToString(ofGetFrameRate()), 10, 20);
    ofDrawBitmapStringHighlight("Tracker thread framerate : "+ofToString(tracker.getThreadFps()), 10, 40);
    
    ofSetColor(255);
    ofDrawBitmapStringHighlight(oscMessageString, 15, ofGetHeight() - 4);
    
    gui.draw();
}
