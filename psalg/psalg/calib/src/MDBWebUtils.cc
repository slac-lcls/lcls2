//-------------------

#include <curl/curl.h>
#include <curl/easy.h>

#include "psalg/utils/Logger.hh" // MSG, LOGGER
#include "psalg/calib/MDBWebUtils.hh"

//using namespace psalg;

namespace psalg {

//-------------------

const std::string request_url_query(const char* url, const char* query) {
  MSG(INFO, "In request_url_query url:" << url << " query:" << query);

  CURL *curl = curl_easy_init();
  char *ct;

  if(curl) {
    //curl_easy_setopt(curl, CURLOPT_URL, "http://example.com");
    curl_easy_setopt(curl, CURLOPT_URL, "https://pswww-dev.slac.stanford.edu/calib_ws"); // .edu:9306
    CURLcode res = curl_easy_perform(curl);
 
    if(CURLE_OK == res) {
      // ask for the content-type
      res = curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &ct);
 
      if((CURLE_OK == res) && ct)
        printf("We received Content-Type: %s\n", ct);
    }
 
    // always cleanup
    curl_easy_cleanup(curl);
  }

  //std::string s = *ct;
  //strcpy(ct, s.c_str());
  return url;
}

//-------------------

const std::vector<char*> database_names(const char* url) {
  return std::vector<char*>();
}

//-------------------

const std::vector<char*> collection_names(const char* dbname, const char* url) {
  return std::vector<char*>();
}

//-------------------

} // namespace calib

//-------------------

