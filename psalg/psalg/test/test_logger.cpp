
#include "psalg/include/Logger.h" // MsgLog

void test_logger_single() {
  std::cout << "In test_logger_single\n";
  Logger::instance()->print();
  LOGPRINT();
  LOGMSG("Hi, this is my test message for logger singleton");
  LOGMSG("Hi, this is my another message for logger singleton");
  LOGGER.print();
}

int main (int argc, char **argv) {
    test_logger_single()
    return EXIT_SUCCESS; //0
}
