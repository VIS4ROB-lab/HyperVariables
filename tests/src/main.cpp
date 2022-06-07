/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <glog/logging.h>
#include <gtest/gtest.h>

int main(int argc, char **argv) {
    // Set logging flags.
    FLAGS_log_dir = "../test/log/";
    FLAGS_logtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = 2;

    // Initialize and run tests.
    google::InitGoogleLogging(argv[0]);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}