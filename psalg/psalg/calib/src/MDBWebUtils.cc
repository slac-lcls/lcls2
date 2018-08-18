//-------------------

#include <iostream> //ostream

#include <curl/curl.h>
#include <curl/easy.h>

#include "psalg/utils/Logger.hh" // MSG, LOGGER
#include "psalg/calib/MDBWebUtils.hh"

// psdaq/pydaq/README.JSON
// psdaq/drp/drp_eval.cc
//

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
//using namespace rapidjson;

//using namespace psalg;

namespace psalg {

  static std::string STR_RESP; //will hold the url's contents
  static rapidjson::Document JSON_DOC;
  std::vector<std::string> LIST_OF_NAMES;

//-------------------

void print_json_value(const rapidjson::Value &value) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    value.Accept(writer);
    std::cout << "In print_json_value" << buffer.GetString() << std::endl;
}

//-------------------

void print_json_doc(const rapidjson::Value &doc, const std::string& offset) {
  // std::cout << "XXX In print_json_doc: isArray: " << doc.IsArray() << '\n';
  const rapidjson::Value& a = doc;
  std::string gap("  ");
  gap += offset;
  if      (a.IsString()) std::cout << gap << " (string) " << a.GetString() << '\n';
  else if (a.IsInt())    std::cout << gap << " (int) " << a.GetInt() << '\n';
  else if (a.IsObject()) std::cout << gap << " (object) TBD\n";
  //else if (a.IsNum())    std::cout << gap << " (number) " << a.GetNum() << '\n';
  else if (a.IsArray()) {
    for (rapidjson::SizeType i = 0; i < a.Size(); i++) {
	std::cout << gap << "[" << i << "]: ";
	print_json_doc(a[i], gap);
    }
  }
  else std::cout << gap << " N/A " << a.GetType() << '\n';
}

//-------------------

void print_vector_of_strings(const std::vector<std::string>& v) {
  std::cout << "In print_vector_of_strings\n";
  for(std::vector<std::string>::const_iterator it  = v.begin(); it != v.end(); it++) 
    std::cout << "  " << *it << '\n';
}

//-------------------

std::string json_doc_to_string(const rapidjson::Document &doc) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    return buffer.GetString();
}

//-------------------

rapidjson::Document chars_to_json_doc(const char* s) {
    rapidjson::Document document;
    document.Parse(s);
    return std::move(document);
}

//-------------------

const std::vector<std::string>& json_doc_to_vector_of_strings(const rapidjson::Document &doc) {
  LIST_OF_NAMES.clear();
  if(! doc.IsArray() || doc.Size()<1) return LIST_OF_NAMES;
  LIST_OF_NAMES.resize(doc.Size());
  LIST_OF_NAMES.clear();
  for (rapidjson::Value::ConstValueIterator itr = doc.Begin(); itr != doc.End(); ++itr) {
    LIST_OF_NAMES.push_back(itr->GetString());
    printf("%s ", itr->GetString());
  }
  printf("\n");
  return LIST_OF_NAMES;
}

//-------------------

size_t string_callback(char* buf, size_t size, size_t nmemb, void* up) {
    //callback must have this declaration
    //buf is a pointer to the data that curl has for us
    //size*nmemb is the size of the buffer
    MSG(DEBUG, "In string_callback size:" << size << " nmemb:" << nmemb);
    size_t nbytes = size*nmemb;
    STR_RESP.clear();
    STR_RESP.resize(nbytes);
    STR_RESP = buf;
    return STR_RESP.size();
    //for (size_t c = 0; c<nbytes; c++) STR_RESP.push_back(buf[c]);
    //return nbytes; //tell curl how many bytes we handled
  }

//-------------------

size_t json_callback(char* buf, size_t size, size_t nmemb, void* up) {
    MSG(DEBUG, "In json_callback size:" << size << " nmemb:" << nmemb);
    JSON_DOC.Parse(buf);
    return size*nmemb; //tell curl how many bytes we handled
  }

//-------------------

const std::string& string_response() { 
  STR_RESP = json_doc_to_string(JSON_DOC);
  return STR_RESP;
}

//-------------------

const rapidjson::Document & json_doc_response() { 
  return JSON_DOC;
}

//-------------------

void request(const char* url, const char* query) {
  MSG(DEBUG, "In request url:" << url << " query:" << query);

  CURL *curl = curl_easy_init();

  if(curl) {
    //curl_easy_setopt(curl, CURLOPT_URL, "http://example.com");
    curl_easy_setopt(curl, CURLOPT_URL, url); // .edu:9306
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &json_callback);
    //curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &string_callback);
    //curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L); //tell curl to output its progress

    CURLcode res = curl_easy_perform(curl);
 
    if(CURLE_OK == res) {
      // ask for the content-type
      char *ct;
      res = curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &ct);

      if((CURLE_OK == res) && ct) {MSG(DEBUG, "Received Content-Type: " << ct);}
      else                         MSG(WARNING, "RESPONSE IS NOT RECEIVED");
    }
    // always cleanup
    curl_easy_cleanup(curl);
    //curl_global_cleanup();

    //MSG(DEBUG, "Exit request");
  }
}

//-------------------

const std::vector<std::string>& database_names(const char* url) {
  request(url);
  //const std::string& s = string_response();
  //std::cout << "XXX resp: " << s << '\n';
  //const rapidjson::Document jdoc = chars_to_json_doc(s.c_str());
  //print_json_value(jdoc); 
  const rapidjson::Document& doc = json_doc_response();
  return json_doc_to_vector_of_strings(doc);
}

//-------------------

const std::vector<std::string>& collection_names(const char* dbname, const char* url) {
  std::string url_tot(url); url_tot += '/'; url_tot += dbname;
  request(url_tot.c_str());
  const rapidjson::Document& doc = json_doc_response();
  return json_doc_to_vector_of_strings(doc);
}

//-------------------

} // namespace calib

//-------------------

