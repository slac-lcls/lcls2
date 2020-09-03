#include <iostream>
#include <libusdusb4.h>

int main() {
  int count = USB4_DeviceCount();

  std::cout << "number of devices: " << count << std::endl;

  return 0;
}

