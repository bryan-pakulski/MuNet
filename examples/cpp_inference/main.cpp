#include <munet/inference.hpp>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: infer MODEL.mnet [vulkan|vulkan:N|cpu]\n";
    return 2;
  }
  try {
    munet::ModelOptions options;
    if (argc == 3) options.device = argv[2];
    munet::Model model(argv[1], options);
    std::cerr << "Device: " << model.device_name() << '\n';
    // Preprocessing belongs to the application. This model takes one row of four features.
    munet::Tensor features{{1, 4}, {1.f, 2.f, 3.f, 4.f}};
    auto outputs = model.run_named({{"features", features}});
    std::cout << std::setprecision(9);
    for (float value : outputs.at("prediction").data) std::cout << value << ' ';
    std::cout << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
