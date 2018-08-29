//-------------------
#include <algorithm>    // std::sort, reverse
#include <iostream> //ostream
#include <array> //array

#include <stdio.h> //sprintf

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

#include <sstream> // std::stringstream

//using namespace psalg;

namespace psalg {

//-------------------

  //static std::string STR_RESP; // holds responce
  static rapidjson::Value EMPTY_VALUE;
  //static rapidjson::Document JSON_DOC;

/*
union Float {
    const float m_float;
    const char  m_bytes[sizeof(float)];
  //uint8_t  m_bytes[sizeof(float)];
};
*/

//-------------------
//-------------------
//-------------------
//-------------------
//-------------------

std::string json_doc_to_string(const rapidjson::Value& v) {
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
  v.Accept(writer);
  return std::move(sb.GetString());
}

//-------------------

void print_json_doc_as_string(const rapidjson::Value& v) {
  std::string s = json_doc_to_string(v);
  std::cout << "In print_json_doc_as_string: " << s << '\n';
}

//-------------------

std::string json_value_type(const rapidjson::Value& v) {
    if     (v.IsString()) return std::string("string");
    else if(v.IsBool())   return std::string("int");
    else if(v.IsNumber()) return std::string("number");
    else if(v.IsInt())    return std::string("int");
    else if(v.IsObject()) return std::string("object");
    else if(v.IsArray())  return std::string("array");
    else {std::string s("Non-implemented type: "); s+=v.GetType();
          return s;
    }
}

//-------------------

void print_json_doc(const rapidjson::Value& doc, const std::string& offset) {
  // std::cout << "XXX In print_json_doc: isArray: " << doc.IsArray() << '\n';
  const rapidjson::Value& v = doc;
  std::string gap("    ");
  gap += offset;
  if     (v.IsString()) std::cout << gap << "(string): " << std::string(v.GetString()).substr(0,1000) << '\n';
  else if(v.IsBool())   std::cout << gap << "(int)   : " << v.GetBool() << '\n';
  else if(v.IsNumber()) std::cout << gap << "(number): " << v.GetInt() << '\n';
  else if(v.IsInt())    std::cout << gap << "(int)   : " << v.GetInt() << '\n';
  else if(v.IsObject()) {
    std::cout << gap << "(object) content:\n";
    for(rapidjson::Value::ConstMemberIterator itr = v.MemberBegin(); itr != v.MemberEnd(); ++itr) {
      std::cout << gap << "  " << itr->name.GetString() << '\n';
      print_json_doc(v[itr->name.GetString()], gap);
    }
  }  
  else if(v.IsArray()) {
    std::cout << gap << "(array) size " << v.Size() << " content:\n";
    for(rapidjson::SizeType i = 0; i < v.Size(); i++) {
      std::cout << gap << "[" << std::setw(3) << i << "]: ";
	print_json_doc(v[i], gap);
    }
  }
  else std::cout << gap << "Non-implemented type: " << v.GetType() << '\n';
}

//-------------------

void print_vector_of_strings(const std::vector<std::string>& v, const char* gap) {
  std::cout << "In print_vector_of_strings\n";
  for(std::vector<std::string>::const_iterator it  = v.begin(); it != v.end(); it++) 
    std::cout << gap << *it << '\n';
}

//-------------------

void chars_to_json_doc(const char* s, rapidjson::Document& jdoc) {
    rapidjson::ParseResult ok = jdoc.Parse(s);
    if (!ok) MSG(ERROR, "JSON parse error: " << ok.Code() << " " << ok.Offset());
}

//-------------------

//rapidjson::Document chars_to_json_doc(const char* s) {
//    rapidjson::Document document;
//    rapidjson::ParseResult ok = document.Parse(s);
//    if (!ok) MSG(ERROR, "JSON parse error: " << ok.Code() << " " << ok.Offset());
//    return std::move(document);
//}

//-------------------

void json_doc_to_vector_of_strings(const rapidjson::Value& doc, std::vector<std::string>& vout) {
  vout.clear();
  if(! doc.IsArray() || doc.Size()<1) return;
  vout.resize(doc.Size());
  vout.clear();
  for (rapidjson::Value::ConstValueIterator itr = doc.Begin(); itr != doc.End(); ++itr) {
    vout.push_back(itr->GetString());
    std::cout <<  itr->GetString();
  }
  std::cout << '\n';
}

//-------------------

/// Returns reference to the static STR_RESP filled out in responce
//const std::string& string_response() { 
  //s = json_doc_to_string(JSON_DOC);
  //  return STR_RESP;
  //}

//-------------------

/// Returns reference to the static JSON_DOC
//const rapidjson::Document& json_doc_response() { 
//  rapidjson::ParseResult ok = JSON_DOC.Parse(STR_RESP.c_str());
//  if (!ok) MSG(ERROR, "JSON parse error: " << ok.Code() << " " << ok.Offset());
//  return JSON_DOC;
//}

//-------------------

/// Parses STR_RESP to the externally defined Document& doc.
void response_to_json_doc(const std::string& sresp, rapidjson::Document& doc) { 
  rapidjson::ParseResult ok = doc.Parse(sresp.c_str());
  // rapidjson::GetParseError_En(ok.Code())
  if (!ok) MSG(ERROR, "JSON parse error: " << ok.Code() << " " << ok.Offset());
}

//-------------------

//Returns string like "https://pswww-dev.slac.stanford.edu//calib_ws/cdb_cspad_0001/cspad_0001?query_string=%7B%22ctype%22%3A+%22pedestals%22%7D"

void string_url_with_query(std::string& url, const char* dbname, const char* colname, const char* query, const char* urlws) {
  url = urlws;
  url += '/';
  url += dbname;
  url += '/';
  url += colname;
  if(query) {
    std::string squery("?query_string=");
    squery += curl_easy_escape(NULL, query, 0);
    url += squery;
    MSG(DEBUG, "string_url_with_query query: "  << query << " escaped: " << squery << " size: " << squery.size());
  }
  MSG(DEBUG, "string_url_with_query url:" << url);
}

//-------------------

size_t _callback(char* buf, size_t size, size_t nmemb, void* pout) {

  //static unsigned counter=0; counter++;

  //callback must have this declaration
  //buf is a pointer to the data that curl has for us
  //MSG(DEBUG, "In _callback size: " << size << " nmemb:" << nmemb);
  //std::cout << "_callback size: " << size << " nmemb:" << nmemb << " buf:" << buf << '\n';
  //std::cout << "_callback call:" << counter << " size: " << size << " nmemb:" << nmemb << " buf:" << std::string(buf).substr(0,20) << "...\n";
  size_t nbytes = size*nmemb; // size of the buffer
  //*((std::string*)pout) += buf;
  ((std::string*)pout)->append((char*)buf, nbytes);
  return nbytes; // STR_RESP.size();
}

//-------------------

// curl -s "https://pswww-dev.slac.stanford.edu/calib_ws/cdb_cspad_0001/cspad_0001?query_string=%7B%22ctype%22%3A+%22pedestals%22%7D"

// perform request with specified url and saves response in static std::string STR_RESP through _callback.

void request(std::string& sresp, const char* url) {
  MSG(DEBUG, "In request url:" << url);

  std::string& _str = sresp; // STR_RESP;

  CURL *curl = curl_easy_init();

  if(curl) {
    //if (LOGGER.logging(LL::DEBUG)) curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
    //curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 240000L); // sets CURL_MAX_READ_SIZE up to max (512kB) = 524288L ?
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &_str);
    curl_easy_setopt(curl, CURLOPT_URL, url);
    //curl_easy_setopt(curl, CURLOPT_TRANSFERTEXT, 1L);

    _str.clear();
    CURLcode res = curl_easy_perform(curl);
    // MSG(DEBUG, "In request _str:" << _str.substr(0,200) << "...\n");

    if(CURLE_OK == res) { // ask for the content-type
      char *ct;
      res = curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &ct);
      if((CURLE_OK == res) && ct) {MSG(DEBUG, "Received Content-Type: " << ct);}
      else                         MSG(WARNING, "RESPONSE IS NOT RECEIVED");
    }

    curl_easy_cleanup(curl); // always cleanup
  }
  MSG(DEBUG, "Exit request");
}

//-------------------

void database_names(std::vector<std::string>& dbnames, const char* urlws) {
  std::string sresp;
  request(sresp, urlws);
  rapidjson::Document doc;
  response_to_json_doc(sresp, doc);
  //print_json_doc_as_string(doc); 
  json_doc_to_vector_of_strings(doc, dbnames);
}

//-------------------

void collection_names(std::vector<std::string>& colnames, const char* dbname, const char* urlws) {
  std::string url(urlws); url += '/'; url += dbname;
  std::string sresp;
  request(sresp, url.c_str());
  rapidjson::Document doc;
  response_to_json_doc(sresp, doc);
  json_doc_to_vector_of_strings(doc, colnames);
}

//-------------------

/// Performs request for url+query and saves responce in static std::string STR_RESP
void find_docs(std::string& sresp, const char* dbname, const char* colname, const char* query, const char* urlws) {
  std::string url;
  string_url_with_query(url, dbname, colname, query, urlws);
  request(sresp, url.c_str());
}

//-------------------

/// Protected return of doc[name] or EMPTY_VALUE if doc[name] is not found.
const rapidjson::Value& value_from_json_doc(const rapidjson::Value& doc, const char* name) {
  rapidjson::Value::ConstMemberIterator itr = doc.FindMember(name);
  return (itr != doc.MemberEnd()) ? itr->value : EMPTY_VALUE;
}

//-------------------

const rapidjson::Value& find_doc(rapidjson::Document& jdoc, const char* dbname, const char* colname, const char* query, const char* urlws) {
  std::string sresp;
  find_docs(sresp, dbname, colname, query, urlws);

  response_to_json_doc(sresp, jdoc);

  if(! jdoc.IsArray()) {
    MSG(WARNING, "find_doc: Unexpected responce structure (is not array)");
    return EMPTY_VALUE;
  }

  if(jdoc.Size()<1) {
    MSG(WARNING, "find_doc: returns array of size 0 for query: " << query);
    return EMPTY_VALUE;
  }

  MSG(DEBUG, "find_doc: received array of size " << jdoc.Size());
  const rapidjson::Value& v = jdoc;

  std::string squery(query);
  const char* key_sort = (squery.find("time_sec") == std::string::npos) ? "run" : "time_sec";
  MSG(DEBUG, "find_doc: key_sort:" << key_sort << '\n');

  std::vector<int> values;
  MSGSTREAM(INFO, out) {
    for(rapidjson::Value::ConstValueIterator itr = v.Begin(); itr != v.End(); ++itr) {
      const rapidjson::Value& d = *itr;
      out << "\n  doc for"  
          << " run: " << std::setw(4) << d["run"].GetInt() 
          << " time_sec: " << d["time_sec"].GetInt() 
          << " time_stamp: " << d["time_stamp"].GetString();
      values.push_back(d[key_sort].GetInt());
    }
  }

  MSGSTREAM(INFO, out) {
    for(std::vector<int>::iterator it=values.begin(); it!=values.end(); ++it) out << ' ' << *it;
  }

  // sort std::vector values
  std::sort(values.begin(), values.end(), [](const int a, const int b){return a > b;});

  MSGSTREAM(INFO, out) {
    for(std::vector<int>::iterator it=values.begin(); it!=values.end(); ++it) out << ' ' << *it;
  }
  
  MSG(DEBUG, "find_doc: key_sort:" << key_sort << " value selected:" << values[0] << '\n');
  
  for(rapidjson::Value::ConstValueIterator itr = v.Begin(); itr != v.End(); ++itr) {
    const rapidjson::Value& d = *itr;
    if(d[key_sort].GetInt() == values[0]) return d;
  }
  return EMPTY_VALUE;
}

//-------------------

// curl -s "https://pswww-dev.slac.stanford.edu/calib_ws/cdb_cspad_0001/cspad_0001/5b6cdde71ead144f115319be"
void get_doc_for_docid(rapidjson::Document& jdoc, const char* dbname, const char* colname, const char* docid, const char* urlws) {
  std::string url(urlws);
  url += '/'; url += dbname; url += '/'; url += colname; url += '/'; url += docid;
  MSG(DEBUG, "get_doc_for_docid url: \"" << url << "\"\n");
  std::string sresp;
  request(sresp, url.c_str());
  response_to_json_doc(sresp, jdoc);
}

//-------------------

// curl -s "https://pswww-dev.slac.stanford.edu/calib_ws/cdb_cspad_0001/gridfs/5b6cdde71ead144f11531999" 
void string_data_for_id(std::string& sresp, const char* dbname, const char* dataid, const char* urlws) {
  std::string url(urlws); url += '/'; url += dbname; url += "/gridfs/"; url += dataid;
  MSG(DEBUG, "get_doc_for_docid url: \"" << url << "\"\n");
  request(sresp, url.c_str());
  //MSG(DEBUG, "XXX RAW data string:\n\n" << sresp.substr(0,200));
}

//-------------------

template<typename TDATA>
void print_string_data_as_array(const std::string& sresp, TDATA& o) { //, const std::string& s) {
  const char* cstr = sresp.c_str();
  const TDATA* arr = reinterpret_cast<const TDATA*>(cstr);
  //size_t size_arr =  sresp.size()/sizeof(TDATA);
  for(const TDATA* p=arr; p<&arr[100]; p++) {
    std::cout << *p << "  ";
  } std::cout << '\n';
}

//-------------------

//template class psalg::int_string_data_as_array<float>; 

template void print_string_data_as_array<int>(const std::string&, int&); 
template void print_string_data_as_array<float>(const std::string&, float&); 
template void print_string_data_as_array<double>(const std::string&, double&); 

//-------------------

} // namespace psalg

//-------------------

