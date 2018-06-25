//-------------------

#include "psalg/include/Logger.h" // MsgLog

//typedef Logger::LEVEL level_t;

//-------------------
//DEBUG, TRACE, INFO, WARNING, ERROR, FATAL, NOLOG
void test_logger() {

  std::cout << "In test_logger\n";

  LOGGER.set_level(Logger::Logger::DEBUG);
  //std::stringstream& out
  LOGGER.logger_info(std::cout); // or Logger::Logger::instance()->logger_info(out);
  //MSG(INFO, out.str());

  int v=345;
  MSGLOG("mylog", DEBUG, "Test MSGLOG" << " this is a test message for logger singleton " << v);
  MSGLOG("mylog", DEBUG, "Test MSGLOG, this is another message for logger singleton");
  MSG(DEBUG,   "Test MSG DEBUG" << " as stream");
  MSG(TRACE,   "Test MSG TRACE");
  MSG(INFO,    "Test MSG INFO");
  MSG(WARNING, "Test MSG WARNING");
  MSG(ERROR,   "Test MSG ERROR");
  MSG(FATAL,   "Test MSG FATAL");
  MSG(NOLOG,   "Test MSG NOLOG");
}

//-------------------

int main (int argc, char **argv) {
  test_logger();
  return EXIT_SUCCESS; //0
}

//-------------------
