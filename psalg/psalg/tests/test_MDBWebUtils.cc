/* 
 * https://curl.haxx.se/libcurl/c/getinfo.html
 */

#include "psalg/utils/Logger.hh" // MSG, LOGGER,...

#include <stdio.h>
#include <iostream>

//#define CURL_STATICLIB
#include <curl/curl.h>
#include <curl/easy.h>

#include "psalg/calib/MDBWebUtils.hh"

using namespace psalg;

//-------------------

void print_hline(const uint nchars, const char c) {printf("%s\n", std::string(nchars,c).c_str());}

//-------------------

int test_getinfo(void) {
  printf("In test_getinfo\n");

  CURL *curl = curl_easy_init();
  if(curl) {
    //curl_easy_setopt(curl, CURLOPT_URL, "http://example.com");
    curl_easy_setopt(curl, CURLOPT_URL, "https://pswww-dev.slac.stanford.edu/calib_ws"); // .edu:9306
    CURLcode res = curl_easy_perform(curl);
 
    if(CURLE_OK == res) {
      char *ct;
      // ask for the content-type
      res = curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &ct);
 
      if((CURLE_OK == res) && ct)
        printf("\nWe received Content-Type: %s\n", ct);
    }
 
    // always cleanup
    curl_easy_cleanup(curl);
  }

  return 0;
}

//-------------------
 
void test_request() {
  printf("In test_request\n");
  request(); // (const char* url=URLWS, const char* query=NONE);
  const std::string& s = string_response();
  std::cout << "XXX resp: " << s << '\n';
}

//-------------------
// returns "https://pswww-dev.slac.stanford.edu//calib_ws/cdb_cspad_0001/cspad_0001?query_string=%7B%22ctype%22%3A+%22pedestals%22%7D"
std::string test_string_url_with_query() {
  printf("In test_string_url_with_query\n");  
  std::string url;
  const char* dbname  = "cdb_cspad_0001";
  const char* colname = "cspad_0001"; 
                        //'{"ctype": "pedestals", "run" : { "$gte":  "85"}}')
  const char* query   =  "{\"ctype\":\"pedestals\", \"run\":{\"$gte\": 85}}";
  //const char* query   =  "{\"ctype\":\"pedestals\"}";
  string_url_with_query(url, dbname, colname, query);
  std::cout << "test_string_url_with_query: " << url << '\n';
  return url;
}

//-------------------
 
// curl -s "https://pswww-dev.slac.stanford.edu//calib_ws/cdb_cspad_0001/cspad_0001?query_string=..."
void test_request_with_query() {
  printf("In test_request_with_query\n");  
  std::string url = test_string_url_with_query(); 
  //request(url.c_str(), "{\"ctype\":\"pedestals\"}");
  request(url.c_str());//, "{\"ctype\":\"pedestals\"}");

  //const rapidjson::Document& doc = json_doc_response();
  rapidjson::Document doc;
  json_doc_response(doc);
  print_json_doc(doc);
}

//-------------------
 
void test_find_docs() {
  printf("In test_find_docs\n");  
  const char* dbname  = "cdb_cspad_0001";
  const char* colname = "cspad_0001"; 
                        //'{"ctype": "pedestals", "run" : { "$gte":  "85"}}')
  const char* query   =  "{\"ctype\":\"pedestals\", \"run\":{\"$gte\": 87}}";
  find_docs(dbname, colname, query);

  const std::string& s = string_response();
  std::cout << "XXX resp: " << s << '\n';
  //rapidjson::Document doc;
  //json_doc_response(doc);
  //print_json_doc(doc);
}

//-------------------
 
void test_database_names() {
  printf("In database_names\n");
  std::vector<std::string> v;
  database_names(v);
  print_vector_of_strings(v);
}

//-------------------
 
void test_collection_names() {
  printf("In test_collection_names\n");
  std::vector<std::string> v;
  collection_names(v, "cdb_cspad_0001");
  print_vector_of_strings(v);
}

//-------------------



//-------------------

int main(void)
{
  //MSG(INFO, LOGGER.tstampStart() << " Logger started"); // optional record
  LOGGER.setLogger(LL::DEBUG, "%H:%M:%S.%f");           // set level and time format
  printf("In test_MDBWebUtils\n");
  print_hline(80,'_');
  //test_getinfo(); print_hline(80,'_');
  test_request(); print_hline(80,'_');
  test_database_names(); print_hline(80,'_');
  test_collection_names(); print_hline(80,'_');
  std::string url =
  test_string_url_with_query(); print_hline(80,'_');
  //test_request_with_query(); print_hline(80,'_');
  test_find_docs(); print_hline(80,'_');
}

//-------------------

