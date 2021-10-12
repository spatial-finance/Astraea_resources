#include "ofApp.h"


//--------------------------------------------------------------
void ofApp::setup(){
    thumbHeight = ofGetHeight() * 0.25;
    margin = 5;
    zoom = 1.25;

    lookupFile = "lookup.json";
    load(lookupFile);
}

//--------------------------------------------------------------
void ofApp::update(){
    
}

//--------------------------------------------------------------
void ofApp::draw(){
    ofBackground(0);
    if (!parsingSuccessful) {
        ofDrawBitmapString("Can't find lookup file: "+lookupFile+"\nSee instructions for how to create one. Or press 'l' to find lookup file.", 50, 50);
        return;
    }

    int centerX;
    float x = 10 - mx;
    float y = thumbHeight * (zoom - 1.0) * 0.5 + 20;
    for (int i=0; i<thumbs.size(); i++) {
        if (x > -thumbs[order[i]].image.getWidth() &&
            x < ofGetWidth()) {
            thumbs[order[i]].image.draw(x, y);
        }
        if (ofGetMouseX() > x && ofGetMouseX() < x+thumbs[order[i]].image.getWidth()) {
            highlighted = order[i];
            centerX = x + 0.5 * thumbs[order[i]].image.getWidth();
        }
        x += (margin + thumbs[order[i]].image.getWidth());
    }
    
    if (highlighted != -1) {
        ofPushStyle();
        ofPushMatrix();
        ofSetRectMode(OF_RECTMODE_CENTER);
        ofSetColor(0, 255, 0);
        ofDrawRectangle(centerX, 0.5 * thumbHeight + y,
                        zoom * thumbs[highlighted].image.getWidth() + 20,
                        zoom * thumbs[highlighted].image.getHeight() + 20);
        ofSetColor(255);
        thumbs[highlighted].image.draw(centerX, 0.5 * thumbHeight + y,
                                       zoom * thumbs[highlighted].image.getWidth(),
                                       zoom * thumbs[highlighted].image.getHeight());
        ofSetColor(0, 200);
        ofDrawRectangle(centerX, zoom * thumbHeight + y - 30, 90, 20);
        ofSetColor(255);
        ofDrawBitmapString("Query Image", centerX - 43, zoom * thumbHeight + y - 22);
        ofPopMatrix();
        ofPopStyle();
        
        float x = margin;
        float y = ofGetHeight() - 2 * (thumbHeight + margin);
        ofSetColor(255);
        ofDrawBitmapString("Nearest neighbor images:", x + margin, y - margin);
        int numNeighbors = min(30, (int) thumbs[highlighted].closest.size()-1);
        for (int i=0; i<numNeighbors; i++) {
            thumbs[thumbs[highlighted].closest[i]].image.draw(x, y);
            x += (margin + thumbs[thumbs[highlighted].closest[i]].image.getWidth());
            if (x > (ofGetWidth() - thumbs[thumbs[highlighted].closest[i+1]].image.getWidth()*0.33)) {
                y += thumbHeight + margin;
                x = margin;
            }
        }
    }
    
    ofSetColor(255);
    ofDrawBitmapString("Drag mouse to scroll images", 4, 12);
    ofDrawBitmapString("Press 'l' to find new lookup file", ofGetWidth()-300, 12);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if (key=='l') {
        ofFileDialogResult result = ofSystemLoadDialog("Which json file to load?", true);
        if (result.bSuccess) {
            load(result.filePath);
        }
    }
}

//--------------------------------------------------------------
void ofApp::load(string lookupFile){
    this->lookupFile = lookupFile;
    fullWidth = 0;
    
    ofJson js;
    ofFile file(lookupFile);
    parsingSuccessful = file.exists();
    
    if (!parsingSuccessful) {
        ofLog(OF_LOG_ERROR) << "parsing not successful";
        return;
    }
    
    thumbs.clear();
    order.clear();

    int idx = 0;
    file >> js;
    for (auto & entry: js) {
        if(!entry.empty()) {
            string path = entry["path"];
            ImageThumb thumb;
            thumb.image.load(path);
            thumb.image.resize(thumbHeight * thumb.image.getWidth() / thumb.image.getHeight(), thumbHeight);
            for (int j=0; j<entry["lookup"].size(); j++) {
                int c = entry["lookup"][j];
                thumb.closest.push_back(c);
            }
            thumbs.push_back(thumb);
            order.push_back(idx);
            fullWidth += (thumb.image.getWidth() + 5);
            idx++;
        }
    }
    random_shuffle(order.begin(), order.end());
    
    mx = fullWidth / 2.0;
    highlighted = -1;
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    float newMx = mx + 10 * (ofGetMouseX() - ofGetPreviousMouseX());
    mx = max(0.0f, min(fullWidth - ofGetWidth(), newMx));
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
