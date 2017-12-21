#!/bin/sh

if [ -f ./minimal_movidius ]; then
    rm ./minimal_movidius
fi

g++ -std=c++11 -g -O0 movidiusdevice.cpp main.cpp -lcrypto -lmvnc -o minimal_movidius
