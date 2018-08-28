//-------------------

#include <iostream> //ostream

#include <curl/curl.h>
#include <curl/easy.h>

#include "psalg/utils/Logger.hh" // MSG, LOGGER
#include "psalg/calib/MDBWebUtils.hh"

// psdaq/pydaq/README.JSON
// psdaq/drp/drp_eval.cc
//
#include <iomanip> // setw
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
//using namespace rapidjson;

//using namespace psalg;

namespace psalg {

  static std::string STR_RESP; // holds responce
  static rapidjson::Document JSON_DOC;
  //std::vector<std::string> LIST_OF_NAMES;

//-------------------

void print_json_value(const rapidjson::Value& value) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    value.Accept(writer);
    std::cout << "In print_json_value" << buffer.GetString() << std::endl;
}

//-------------------

void print_json_doc(const rapidjson::Value& doc, const std::string& offset) {
  // std::cout << "XXX In print_json_doc: isArray: " << doc.IsArray() << '\n';
  const rapidjson::Value& a = doc;
  std::string gap("    ");
  gap += offset;
  if      (a.IsString()) std::cout << gap << " (string): " << a.GetString() << '\n';
  else if (a.IsBool())   std::cout << gap << " (int)   : " << a.GetBool() << '\n';
  else if (a.IsNumber()) std::cout << gap << " (number): " << a.GetInt() << '\n';
  else if (a.IsInt())    std::cout << gap << " (int)   : " << a.GetInt() << '\n';
  else if (a.IsObject()) {
    for (rapidjson::Value::ConstMemberIterator itr = a.MemberBegin(); itr != a.MemberEnd(); ++itr) {
      std::cout << gap << " (object): " << itr->name.GetString() << '\n';
      print_json_doc(a[itr->name.GetString()], gap);
    }
  }  
  else if (a.IsArray()) {
    for (rapidjson::SizeType i = 0; i < a.Size(); i++) {
      std::cout << gap << "[" << std::setw(3) << i << "]: ";
	print_json_doc(a[i], gap);
    }
  }
  else std::cout << gap << " N/A " << a.GetType() << '\n';
}

//-------------------

void print_vector_of_strings(const std::vector<std::string>& v, const char* gap) {
  std::cout << "In print_vector_of_strings\n";
  for(std::vector<std::string>::const_iterator it  = v.begin(); it != v.end(); it++) 
    std::cout << gap << *it << '\n';
}

//-------------------

std::string json_doc_to_string(const rapidjson::Document& doc) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    return buffer.GetString();
}

//-------------------

rapidjson::Document chars_to_json_doc(const char* s) {
    rapidjson::Document document;
    rapidjson::ParseResult ok = document.Parse(s);
    if (!ok) MSG(ERROR, "JSON parse error: " << ok.Code() << " " << ok.Offset());
    return std::move(document);
}

//-------------------

void json_doc_to_vector_of_strings(const rapidjson::Document& doc, std::vector<std::string>& vout) {
  vout.clear();
  if(! doc.IsArray() || doc.Size()<1) return;
  vout.resize(doc.Size());
  vout.clear();
  for (rapidjson::Value::ConstValueIterator itr = doc.Begin(); itr != doc.End(); ++itr) {
    vout.push_back(itr->GetString());
    printf("%s ", itr->GetString());
  }
  printf("\n");
}

//-------------------

size_t string_callback(char* buf, size_t size, size_t nmemb, void* up) {
    //callback must have this declaration
    //buf is a pointer to the data that curl has for us
    MSG(DEBUG, "In string_callback size:" << size << " nmemb:" << nmemb);
    size_t nbytes = size*nmemb; // size of the buffer
    STR_RESP += buf;
    return nbytes; // STR_RESP.size();
}

//-------------------

/// Returns reference to the static STR_RESP filled out in responce
const std::string& string_response() { 
  //s = json_doc_to_string(JSON_DOC);
  return STR_RESP;
}

//-------------------

/// Returns reference to the static JSON_DOC
const rapidjson::Document& json_doc_response() { 
  rapidjson::ParseResult ok = JSON_DOC.Parse(STR_RESP.c_str());
  if (!ok) MSG(ERROR, "JSON parse error: " << ok.Code() << " " << ok.Offset());
  return JSON_DOC;
}

//-------------------

/// Parses STR_RESP to the externally defined Document& doc.
void json_doc_response(rapidjson::Document& doc) { 
  rapidjson::ParseResult ok = doc.Parse(STR_RESP.c_str());
  // rapidjson::GetParseError_En(ok.Code())
  if (!ok) MSG(ERROR, "JSON parse error: " << ok.Code() << " " << ok.Offset());
}

//-------------------

//Returns string like "https://pswww-dev.slac.stanford.edu//calib_ws/cdb_cspad_0001/cspad_0001?query_string=%7B%22ctype%22%3A+%22pedestals%22%7D"

void string_url_with_query(std::string& surl, const char* dbname, const char* colname, const char* query, const char* urlws) {
  surl = urlws;
  surl += '/';
  surl += dbname;
  surl += '/';
  surl += colname;
  MSG(DEBUG, "string_url_with_query url:" << surl << " query:");

  if(query) {
    std::string squery("?query_string=");
    squery += curl_easy_escape(NULL, query, 0);
    MSG(DEBUG, "query: "  << query << " escaped: " << squery << " size: " << squery.size());
    surl += squery;
    MSG(DEBUG, "url + query: "  << surl);
  }
}

//-------------------
// curl -s "https://pswww-dev.slac.stanford.edu//calib_ws/cdb_cspad_0001/cspad_0001?query_string=%7B%22ctype%22%3A+%22pedestals%22%7D"

// perform request with specified url and seves response in static std::string STR_RESP through string_callback.

void request(const char* url) {
  MSG(DEBUG, "In request url:" << url);

  CURL *curl = curl_easy_init();

  if(curl) {
    //if (LOGGER.logging(LL::DEBUG)) curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &string_callback);
    curl_easy_setopt(curl, CURLOPT_URL, url);

    STR_RESP.clear();
    CURLcode res = curl_easy_perform(curl);
    MSG(DEBUG, "In request STR_RESP:" << STR_RESP << '\n');

    if(CURLE_OK == res) { // ask for the content-type
      char *ct;
      res = curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &ct);
      if((CURLE_OK == res) && ct) {MSG(DEBUG, "Received Content-Type: " << ct);}
      else                         MSG(WARNING, "RESPONSE IS NOT RECEIVED");
    }

    curl_easy_cleanup(curl); // always cleanup
  }
}

//-------------------

void database_names(std::vector<std::string>& dbnames, const char* urlws) {
  request(urlws);
  rapidjson::Document doc;
  json_doc_response(doc);
  //print_json_value(doc); 
  json_doc_to_vector_of_strings(doc, dbnames);
}

//-------------------

void collection_names(std::vector<std::string>& colnames, const char* dbname, const char* urlws) {
  std::string url_tot(urlws); url_tot += '/'; url_tot += dbname;
  request(url_tot.c_str());
  rapidjson::Document doc;
  json_doc_response(doc);
  json_doc_to_vector_of_strings(doc, colnames);
}

//-------------------

void find_docs(const char* dbname, const char* colname, const char* query, const char* urlws) {
  std::string surl;
  string_url_with_query(surl, dbname, colname, query, urlws);
  request(surl.c_str());
}

//-------------------
// curl -s "https://pswww-dev.slac.stanford.edu/calib_ws/cdb_cxic0415/cspad_0001/5b6893e81ead141643fe4344"
//void get_doc_for_docid(dbname, colname, docid, urlws=cc.URLWS) :

// curl -s "https://pswww-dev.slac.stanford.edu/calib_ws/cdb_cxic0415/gridfs/5b6893d91ead141643fe3f6a" 
//void get_data_for_id(dbname, dataid, urlws=cc.URLWS) :

//-------------------

} // namespace calib

//-------------------

