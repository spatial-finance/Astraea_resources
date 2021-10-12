#include "ofApp.h"


void ofApp::setup() {
    ofSetWindowShape(540, 360);
    ofSetWindowTitle("ConvnetOSC");
    
#ifdef RELEASE
    ccv.setup(ofToDataPath("image-net-2012.sqlite3"));
#else
    ccv.setup(ofToDataPath("../../../../data/image-net-2012.sqlite3"));
#endif
    
    if (!ccv.isLoaded()) return;
    
    // default settings
    oscDestination = OSC_DESTINATION_DEFAULT;
    oscAddressRoot = OSC_ADDRESS_ROOT_DEFAULT;
    oscPort = OSC_PORT_DEFAULT;
    deviceId = DEVICE_ID_DEFAULT;
    
    // load settings from file
    ofXml xml;
    xml.load("settings_convnet.xml");
    
    oscDestination = xml.getChild("ConvnetOSC").getChild("ip").getValue();
    oscPort = ofToInt(xml.getChild("ConvnetOSC").getChild("port").getValue());
    oscAddressRoot = xml.getChild("ConvnetOSC").getChild("address").getValue();
    bool sendClassificationsByDefault = (xml.getChild("ConvnetOSC").getChild("sendClassificationsByDefault").getValue() == "1");
    deviceId = ofToInt(xml.getChild("ConvnetOSC").getChild("deviceId").getValue());
    
    cam.setDeviceID(deviceId);
    cam.initGrabber(320, 240);
    
    // setup osc
    osc.setup(oscDestination, oscPort);
    sending = false;
    
    // setup gui
    gui.setup();
    gui.setName("ConvnetOSC");
    gui.add(sending.set("sending", false));
    gui.add(sendClassifications.setup("send classifications", sendClassificationsByDefault));
}

void ofApp::update() {
    if (!ccv.isLoaded()) return;
    cam.update();
    if (cam.isFrameNew() && sending) {
        sendOsc();
    }
}

void ofApp::sendOsc() {
    int layer = sendClassifications ? ccv.numLayers() : ccv.numLayers()-1;
    featureEncoding = ccv.encode(cam, layer);
    
    msg.clear();
    msg.setAddress(oscAddressRoot);
    for (int i=0; i<featureEncoding.size(); i++) {
        msg.addFloatArg(featureEncoding[i]);
    }
    osc.sendMessage(msg);
}

void ofApp::keyPressed(int key) {
    if (key==' ') {
        sending = !sending;
    }
}

void ofApp::draw() {
    if (!ccv.isLoaded()) {
        ofPushStyle();
        ofSetColor(ofColor::red);
        ofDrawBitmapString("Network file not found!\nCheck your data folder to make sure it exists.", 20, 20);
        ofPopStyle();
        return;
    }
    
    if (sending) {
        ofBackground(0, 255, 0);
        
        ofSetColor(0, 100);
        ofDrawRectangle(10, 280, 520, 66);
        ofSetColor(255);
        string txt = "sending "+ofToString(sendClassifications ? "classification probabilities":"fc2 layer activations")+" ("+ofToString(msg.getNumArgs())+" values)\n";
        txt += "to "+oscDestination+", port "+ofToString(oscPort)+",\n";
        txt += "osc address \""+oscAddressRoot+"\"\n";
        txt += "press spacebar or click 'sending' to turn off sending.";
        ofDrawBitmapString(txt, 20, 296);
    }
    else {
        ofBackground(255, 0, 0);
        
        ofSetColor(0, 100);
        ofDrawRectangle(10, 280, 520, 66);
        ofSetColor(255);
        string txt = "press spacebar or click 'sending' to turn on sending.";
        ofDrawBitmapString(txt, 20, 296);
    }
    
    ofSetColor(255);
    cam.draw(210, 10);
    gui.draw();

}
