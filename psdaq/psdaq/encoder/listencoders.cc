#include <iostream>
#include <libusdusb4.h>

int main() {
  std::cout << "Initializing device" << std::endl;
  short deviceCount;
  int result = USB4_Initialize(&deviceCount);
  if (result != USB4_SUCCESS) {
    std::cout << "Failed to initialize USB4 driver (" << result << ")" << std::endl;
  } else {
    std::cout << "Found " << deviceCount << " device" << (deviceCount==1 ? "" : "s") << "!" << std::endl;
    for(short i=0; i<deviceCount; i++) {
      unsigned short version = 0;
      unsigned short model = 0;
      unsigned long serial_num = 0;
      unsigned char month = 0;
      unsigned char day = 0;
      unsigned short year = 0;

      std::cout << "Device 0:" << std::endl;

      result = USB4_GetFactoryInfo(i, &model, &version, &serial_num, &month, &day, &year);
      if (result != USB4_SUCCESS) {
        std::cout << "  USB4_GetFactoryInfo return error code: " << result << std::endl;
      } else {
        std::cout << "  Model        : " << model << std::endl;
        std::cout << "  Version      : " << version << std::endl;
        std::cout << "  Serial Number: " << std::hex << serial_num << std::endl;
        std::cout.copyfmt(std::ios(NULL));
        std::cout << "  Date         : " << static_cast<unsigned>(month) << "/"
                                         << static_cast<unsigned>(day)   << "/"
                                         << year << std::endl;
      }
    }
  }

  std::cout << "Shutting down device" << std::endl;
  USB4_Shutdown();

  return 0;
}

