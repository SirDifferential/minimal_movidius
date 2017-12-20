#!/bin/sh

g++ -std=c++11 -g -O0 movidiusdevice.cpp main.cpp -lcrypto -lmvnc -o minimal_movidius
