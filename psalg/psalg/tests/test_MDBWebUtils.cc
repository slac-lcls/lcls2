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
 
void test_request() {
  printf("In test_request\n");
  std::string sresp;
  request(sresp, URLWS);
  std::cout << "XXX url:" << URLWS << "\n   resp: " << sresp << '\n';
}

//-------------------
// returns "https://pswww-dev.slac.stanford.edu//calib_ws/cdb_cspad_0001/cspad_0001?query_string=%7B%22ctype%22%3A+%22pedestals%22%7D"
std::string test_string_url_with_query() {
  printf("In test_string_url_with_query\n");  
  std::string url;
  const char* dbname  = "cdb_cspad_0001";
  const char* colname = "cspad_0001"; 
  const char* query   = "{\"ctype\":\"pedestals\", \"run\":{\"$gte\": 87}}";
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
  std::string sresp;
  request(sresp, url.c_str());
  rapidjson::Document doc;
  response_to_json_doc(sresp, doc);
  print_json_doc(doc);
}

//-------------------
 
void test_find_docs() {
  printf("In test_find_docs\n");  
  const char* dbname  = "cdb_cspad_0001";
  const char* colname = "cspad_0001"; 
  const char* query   =  "{\"ctype\":\"pedestals\", \"run\":{\"$gte\": 87}}"; // 2 db docs
  //const char* query   =  "{\"ctype\":\"pedestals\", \"run\":78}";           // 1 db doc
  //const char* query   =  "{\"ctype\":\"pedestals\", \"run\":1234}";         // 0 db docs
  //const char* query   =  "{\"ctype\":\"pedestals\"}";                       // many db docs
  std::string sresp;
  find_docs(sresp, dbname, colname, query);
  std::cout << "XXX string response: " << sresp << '\n';

  rapidjson::Document doc;
  response_to_json_doc(sresp, doc);
  print_json_doc(doc);
}

//-------------------
 
void test_find_doc() {
  printf("In test_find_doc\n");  
  const char* dbname  = "cdb_cspad_0001";
  const char* colname = "cspad_0001"; 
  //const char* query   =  "{\"ctype\":\"pedestals\", \"run\":{\"$gte\": 87}}"; // 2 db docs
  //const char* query   =  "{\"ctype\":\"pedestals\", \"run\":78}";           // 1 db doc
  //const char* query   =  "{\"ctype\":\"pedestals\", \"run\":1234}";         // 0 db docs
  //const char* query   =  "{\"ctype\":\"pedestals\"}";                       // many db docs
  const char* query   =  "{\"ctype\":\"geometry\"}";                       // many db docs
  rapidjson::Document jdoc;
  const rapidjson::Value& jv = find_doc(jdoc, dbname, colname, query);
  std::string s = json_doc_to_string(jv);
  std::cout << "XXX string content of found doc: " << s << '\n';
  // "id_data":"5b6cdde71ead144f11531999","data_type":"ndarray","data_dtype":"float32","data_size":"2296960","data_ndim":"2","data_shape":"(5920, 388)"
  /*
  std::cout << "\ndoc metadata:"
            << "\n id_data    : " << jv["id_data"].GetString()
            << "\n data_type  : " << jv["data_type"].GetString()
            << "\n data_dtype : " << jv["data_dtype"].GetString()
            << "\n data_size  : " << jv["data_size"].GetString()
            << "\n data_ndim  : " << jv["data_ndim"].GetString()
            << "\n data_shape : " << jv["data_shape"].GetString()
            << '\n';
  */
}

//-------------------

void test_get_doc_for_docid() {
  printf("In test_get_doc_for_docid\n");  
  rapidjson::Document jdoc;
  const char* dbname  = "cdb_cspad_0001";
  const char* colname = "cspad_0001"; 
  const char* docid   = "5b6cdde71ead144f115319be";
  get_doc_for_docid(jdoc, dbname, colname, docid);
  std::string s = json_doc_to_string(jdoc);
  std::cout << "XXX string content of found doc: " << s << '\n';
}

//-------------------

void test_get_data_for_id() {

  float v;

  const char* dbname1 = "cdb_cxid9114";
  const char* dataid1 = "5b6cdde71ead144f11531974";
  std::cout << "In test_get_data_for_id PEDESTALS for " << dbname1 << " and " << dataid1 << ":\n";  
  std::string sresp;
  string_data_for_id(sresp, dbname1, dataid1);
  print_string_data_as_array<float>(sresp, v);

  print_hline(80,'_');
  const char* dbname2 = "cdb_cspad_0001";
  const char* dataid2 = "5b6cdde71ead144f11531999";
  std::cout << "In test_get_data_for_id PEDESTALS for " << dbname2 << " and " << dataid2 << ":\n";  
  string_data_for_id(sresp, dbname2, dataid2);
  print_string_data_as_array<float>(sresp, v);

  print_hline(80,'_');
  const char* dbname3 = "cdb_cspad_0001";
  const char* dataid3 = "5b6cdef31ead1451fc5f0a2e"; // for geometry
  std::cout << "In test_get_data_for_id GEOMETRY for " << dbname3 << " and " << dataid3 << ":\n";  
  string_data_for_id(sresp, dbname3, dataid3);
  std::cout << sresp << '\n';

  //std::string s = json_doc_to_string(jdoc);
  //std::cout << "XXX string content of found data: " << s.substr(0,100) << '\n';
  //print_json_doc(jdoc);
  //std::cout << "XXX test_get_data_for_id: " << json_value_type(jdoc) << '\n';
}

//-------------------

int main(void)
{
  //MSG(INFO, LOGGER.tstampStart() << " Logger started"); // optional record
  LOGGER.setLogger(LL::DEBUG, "%H:%M:%S.%f");           // set level and time format
  printf("In test_MDBWebUtils\n");
  print_hline(80,'_');
  test_request(); print_hline(80,'_');
  test_database_names(); print_hline(80,'_');
  test_collection_names(); print_hline(80,'_');
  std::string url =
  test_string_url_with_query(); print_hline(80,'_');
  //test_request_with_query(); print_hline(80,'_');
  test_find_docs(); print_hline(80,'_');
  test_find_doc(); print_hline(80,'_');

  test_get_doc_for_docid(); print_hline(80,'_');
  test_get_data_for_id(); print_hline(80,'_');
}

//-------------------

