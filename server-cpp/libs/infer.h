#ifndef INFER_H // To make sure you don't declare the function more than once by including the header multiple times.
#define INFER_H

#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <memory>

#include <torch/script.h>

#include <opencv2/core/core.hpp>

#include "torchutils.h"
#include "opencvutils.h"

std::tuple<std::string, std::string> infer(cv::Mat, int, int, std::vector<double>, std::vector<double>, std::vector<std::string>, torch::jit::script::Module, bool);

#endif
