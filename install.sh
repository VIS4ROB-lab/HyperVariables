#!/bin/sh

sudo apt-get update
sudo apt-get install libgflags-dev libeigen3-dev

mkdir install && cd install || exit
wget -O gtest.tar.gz https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz
wget -O glog.tar.gz https://github.com/google/glog/archive/refs/tags/v0.6.0.tar.gz

tar -xzf gtest.tar.gz
tar -xzf glog.tar.gz

mkdir gtest
mkdir glog

tar -xzf gtest.tar.gz -C gtest --strip-components=1
tar -xzf glog.tar.gz -C glog --strip-components=1

mkdir gtest/build
cmake -S gtest -B gtest/build
sudo make install -C gtest/build -j$(nproc)

mkdir glog/build
cmake -S glog -B glog/build
sudo make install -C glog/build -j$(nproc)
