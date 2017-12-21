Minimal example showing some age and gender detection using the caffe networks with movidius

Build using compile.sh or `g++ -std=c++11 -g -O0 movidiusdevice.cpp main.cpp -lcrypto -lmvnc -o minimal_movidius`
The networks are here http://plantmonster.net/koodailut/movidius/network.zip (They are simply the Age and Gender caffe networks built with MVNCCompile)
