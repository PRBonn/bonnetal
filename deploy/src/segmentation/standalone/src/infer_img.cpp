/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of Bonnetal, and covered by the provided LICENSE file.
 *
 */

// opencv stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// c++ stuff
#include <chrono>
#include <iomanip>  // for setfill
#include <iostream>
#include <string>

// net stuff
#include <selector.hpp>
namespace cl = bonnetal::segmentation;

// standalone lib h
#include "infer.hpp"

// boost
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

int main(int argc, const char *argv[]) {
  // define options
  std::vector<std::string> images;
  std::string path;
  std::string backend = "pytorch";
  bool verbose = false;

  // Parse options
  try {
    po::options_description desc{"Options"};
    desc.add_options()("help,h", "Help screen")(
        "image,i", po::value<std::vector<std::string>>(&images)->multitoken(),
        "Images to infer. No Default")(
        "path,p", po::value<std::string>(),
        "Directory to get the inference model from. No default")(
        "backend,b", po::value<std::string>(),
        "Backend. Pytorch, and TensorRT.")(
        "verbose,v", po::bool_switch(),
        "Verbose mode. Calculates profile (time to run)");

    po::variables_map vm;
    po::store(parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }

    if (!vm["image"].empty()) {
      for (const auto &image : images) {
        std::cout << "image: " << image << std::endl;
      }
    } else {
      std::cerr << "No images! See --help (-h) for help. Exiting" << std::endl;
      return 1;
    }

    // make defaults count, parameter check, and print
    if (vm.count("path")) {
      path = vm["path"].as<std::string>() + "/";  // make sure path is valid
      std::cout << "path: " << path << std::endl;
    } else {
      std::cerr << "No path! See --help (-h) for help. Exiting" << std::endl;
      return 1;
    }
    if (vm.count("backend")) {
      backend = vm["backend"].as<std::string>();
      std::cout << "backend: " << backend << std::endl;
    } else {
      std::cout << "backend: " << backend << ". Using default!" << std::endl;
    }
    if (vm.count("verbose")) {
      verbose = vm["verbose"].as<bool>();
      std::cout << "verbose: " << verbose << std::endl;
    } else {
      std::cout << "verbose: " << verbose << ". Using default!" << std::endl;
    }

    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
  } catch (const po::error &ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }

  // create a network
  std::unique_ptr<cl::Net> net = cl::make_net(path, backend);

  // set verbosity
  net->verbosity(verbose);

  if (verbose) {
    cv::namedWindow("Frame", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Mask", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Blend", cv::WINDOW_AUTOSIZE);
  }

  // predict each image
  for (auto image : images) {
    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
    std::cout << "Predicting image: " << image << std::endl;

    // Open an image
    cv::Mat cv_img = cv::imread(image);
    // Check for invalid input
    if (!cv_img.data) {
      std::cerr << "Could not open or find the image" << std::endl;
      return 1;
    }

    // predict
    cv::Mat argmax = net->infer(cv_img);

    // get color
    cv::Mat color_mask = net->color(argmax);

    // get color
    cv::Mat blend_mask = net->blend(cv_img, color_mask);

    // print the output
    if (verbose) {
      cv::imshow("Frame", cv_img);      // Show our image inside
      cv::imshow("Mask", color_mask);   // Show our image inside
      cv::imshow("Blend", blend_mask);  // Show our image inside
      cv::waitKey(0);
    }
    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
  }

  return 0;
}
