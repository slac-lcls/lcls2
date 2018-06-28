//-------------------

#include "psalg/utils/Logger.hh" // MSG, LOGGER,...

//typedef Logger::LEVEL level_t;

//-------------------
//DEBUG, TRACE, INFO, WARNING, ERROR, FATAL, NOLOG
void test_logger() {

  //LOGGER.setLevel(LL::DEBUG);
  // LOGGER.setTimeFormat("%Y-%m-%d %H:%M:%S.%f");
  //std::stringstream& out
  //LOGGER.loggerInfo(std::cout); // or Logger::Logger::instance()->loggerIinfo(out);
  //MSG(INFO, out.str());
  MSG(INFO, LOGGER.tstampStart() << " Logger started"); // optional record
  LOGGER.setLogger(LL::DEBUG, "%H:%M:%S.%f");           // set level and time format

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
  MSGSTREAM(INFO, out){out << "Test MSGSTREAM INFO"; out << " which supports block of output statements as a single message";} 
  MSG(DEBUG,    "Start logger on " << LOGGER.tstampStart());

  LOGGER.setTimeFormat("%Y-%m-%d %H:%M:%S.%f");
  MSG(INFO, "Test setTimeFormat");

  LOGGER.setTimeFormat("");
  MSG(INFO, "Test setTimeFormat(empty)");
}

//-------------------

int main (int argc, char **argv) {
  test_logger();
  return EXIT_SUCCESS; //0
}

//-------------------
