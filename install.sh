#!/bin/sh
sudo apt-get update
sudo apt-get install libgflags-dev libeigen3-dev

mkdir install && cd install || exit
wget -O gtest.tar.gz https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz
wget -O glog.tar.gz https://github.com/google/glog/archive/refs/tags/v0.6.0.tar.gz

tar xzf gtest.tar.gz
tar xzf glog.tar.gz

mkdir gtest/build
cmake -B ./gtest/build -S ./gtest && make -I ./gtest/build -j$(nproc)
sudo make install -I ./gtest/build