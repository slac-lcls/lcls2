//-------------------
#include <algorithm>    // std::sort, reverse, replace
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
  //static rapidjson::Document JSON_DOC;

//-------------------
//-------------------
//-------------------
//-------------------
//-------------------
/// Converts rapidjson Document or Value object in string.

std::string json_doc_to_string(const rapidjson::Value& v) {
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
  v.Accept(writer);
  return std::move(sb.GetString());
}

//-------------------
/// Prints rapidjson Document or Value.

void print_json_doc_as_string(const rapidjson::Value& v) {
  std::string s = json_doc_to_string(v);
  std::cout << "In print_json_doc_as_string: " << s << '\n';
}

//-------------------
/// Returns string with rapidjson value type name.

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
/// Recursively prints content of the rapidjson Document or Value.

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
/// Prints vector of strings.

void print_vector_of_strings(const std::vector<std::string>& v, const char* gap) {
  std::cout << "In print_vector_of_strings\n";
  for(std::vector<std::string>::const_iterator it  = v.begin(); it != v.end(); it++) 
    std::cout << gap << *it << '\n';
}

//-------------------
/// Prints float array of values from input byte-string.

void print_byte_string_as_float(const std::string& s, const size_t nvals) {
  typedef float T;
  const T* pout;
  size_t size;
  response_string_to_data_array<T>(s, pout, size);
  std::cout << "data size: " << size << '\n';
  for(const float* p=pout; p<&pout[nvals]; p++) {std::cout << *p << "  ";} std::cout << '\n';
}

//-------------------
/// Parses char* to Document& doc.

void chars_to_json_doc(const char* s, rapidjson::Document& jdoc) {
    rapidjson::ParseResult ok = jdoc.Parse(s);
    if (!ok) MSG(ERROR, "JSON parse error: " << ok.Code() << " " << ok.Offset());
}

//-------------------
/// Converts response byte-string to rapidjson::Document.

void response_to_json_doc(const std::string& sresp, rapidjson::Document& jdoc) { 
  chars_to_json_doc(sresp.c_str(), jdoc);
}

//-------------------
/// Converts rapidjson::Value (for IsArray()) to the std::vector of string names.

void json_doc_to_vector_of_strings(const rapidjson::Value& jdoc, std::vector<std::string>& vout) {
  vout.clear();
  if(! jdoc.IsArray() || jdoc.Size()<1) return;
  vout.resize(jdoc.Size());
  vout.clear();
  for (rapidjson::Value::ConstValueIterator itr = jdoc.Begin(); itr != jdoc.End(); ++itr) {
    vout.push_back(itr->GetString());
  }
}
//-------------------
/// Returns pointer TDATA* to array of values and its size associated with byte-string.

template<typename TDATA>
void response_string_to_data_array(const std::string& sresp, const TDATA*& pout, size_t& size) {
  const char* cstr = sresp.c_str();
  pout = reinterpret_cast<const TDATA*>(cstr);
  size = sresp.size()/sizeof(TDATA);
  //std::cout << "==== XXX In response_string_to_data_array data size: " << size << '\n';
  //for(const TDATA* p=pout; p<&pout[100]; p++) {std::cout << *p << "  ";} std::cout << "\n====\n";
}

//-------------------
/// Combines urlws, dbname, colname, and query in an output string url.
/// Return e.g. "https://pswww.slac.stanford.edu/calib_ws/cdb_cspad_0001/cspad_0001?query_string=%7B%22ctype%22%3A+%22pedestals%22%7D"

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
/// callback for curl_easy.

size_t _callback(char* buf, size_t size, size_t nmemb, void* pout) {
  size_t nbytes = size*nmemb; // size of the buffer
  //*((std::string*)pout) += buf;
  ((std::string*)pout)->append((char*)buf, nbytes);
  return nbytes;
}

//-------------------
/// perform request with specified url and saves response in std::string& sresp through _callback. Analog of
/// curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_cspad_0001/cspad_0001?query_string=%7B%22ctype%22%3A+%22pedestals%22%7D"

void request(std::string& sresp, const char* url) {
  MSG(DEBUG, "In request url:" << url);

  CURL *curl = curl_easy_init();

  if(curl) {
    //if (LOGGER.logging(LL::DEBUG)) curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
    //curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 240000L); // sets CURL_MAX_READ_SIZE up to max (512kB) = 524288L ?
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &sresp);
    curl_easy_setopt(curl, CURLOPT_URL, url);
    //curl_easy_setopt(curl, CURLOPT_TRANSFERTEXT, 1L);

    sresp.clear();
    CURLcode res = curl_easy_perform(curl);
    // MSG(DEBUG, "In request sresp:" << sresp.substr(0,200) << "...\n");

    if(CURLE_OK == res) { // ask for the content-type
      char *ct;
      res = curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &ct);
      if((CURLE_OK == res) && ct) {MSG(DEBUG, "Received Content-Type: " << ct << " sresp.size=" << sresp.size());}
      else                         MSG(WARNING, "RESPONSE IS NOT RECEIVED");
    }

    curl_easy_cleanup(curl); // always cleanup
  }
  MSG(DEBUG, "Exit request");
}

//-------------------
/// Returns database names for default or specified mongod server.

void database_names(std::vector<std::string>& dbnames, const char* urlws) {
  std::string sresp;
  request(sresp, urlws);
  rapidjson::Document jdoc;
  response_to_json_doc(sresp, jdoc);
  json_doc_to_vector_of_strings(jdoc, dbnames);
}

//-------------------
/// Returns collection names for default or specified database on mongod server.

void collection_names(std::vector<std::string>& colnames, const char* dbname, const char* urlws) {
  std::string url(urlws); url += '/'; url += dbname;
  std::string sresp;
  request(sresp, url.c_str());
  rapidjson::Document jdoc;
  response_to_json_doc(sresp, jdoc);
  json_doc_to_vector_of_strings(jdoc, colnames);
}

//-------------------
/// Performs request for url+query and saves responce in std::string& sresp

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
/// Returns a single document "best-latest-matched" to specified in query conditions.

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
  MSG(DEBUG, "find_doc: key_sort:" << key_sort);

  std::vector<int> values;
  for(rapidjson::Value::ConstValueIterator itr = v.Begin(); itr != v.End(); ++itr) {
     values.push_back((*itr)[key_sort].GetInt());
  }

  MSGSTREAM(DEBUG, out) {
    for(rapidjson::Value::ConstValueIterator itr = v.Begin(); itr != v.End(); ++itr) {
      const rapidjson::Value& d = *itr;
      out << "\n  doc for"  
          << " run: " << std::setw(4) << d["run"].GetInt() 
          << " time_sec: " << d["time_sec"].GetInt() 
          << " time_stamp: " << d["time_stamp"].GetString();
    }
  }

  MSGSTREAM(DEBUG, out) {
    out << "unsorted:";
    for(std::vector<int>::iterator it=values.begin(); it!=values.end(); ++it) out << ' ' << *it;
  }

  // sort std::vector values
  std::sort(values.begin(), values.end(), [](const int a, const int b){return a > b;});

  MSGSTREAM(DEBUG, out) {
    out << "  sorted:";
    for(std::vector<int>::iterator it=values.begin(); it!=values.end(); ++it) out << ' ' << *it;
    out << "\n find_doc: key_sort:" << key_sort << " value selected:" << values[0];
  }
  
  for(rapidjson::Value::ConstValueIterator itr = v.Begin(); itr != v.End(); ++itr) {
    const rapidjson::Value& d = *itr;
    if(d[key_sort].GetInt() == values[0]) return d;
  }
  return EMPTY_VALUE;
}

//-------------------
/// Returns document for specified document id from database collection. Analog of
/// curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_cspad_0001/cspad_0001/5b6cdde71ead144f115319be"

void get_doc_for_docid(rapidjson::Document& jdoc, const char* dbname, const char* colname, const char* docid, const char* urlws) {
  std::string url(urlws);
  url += '/'; url += dbname; url += '/'; url += colname; url += '/'; url += docid;
  MSG(DEBUG, "get_doc_for_docid url: \"" << url << "\"");
  std::string sresp;
  request(sresp, url.c_str());
  response_to_json_doc(sresp, jdoc);
}

//-------------------
/// Returns byte-string data for specified dataid from database GridFS. Analog of
/// curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_cspad_0001/gridfs/5b6cdde71ead144f11531999" 

void get_data_for_id(std::string& sresp, const char* dbname, const char* dataid, const char* urlws) {
  std::string url(urlws); url += '/'; url += dbname; url += "/gridfs/"; url += dataid;
  MSG(DEBUG, "get_doc_for_docid url: \"" << url << "\"");
  request(sresp, url.c_str());
  //MSG(DEBUG, "XXX RAW data string:\n\n" << sresp.substr(0,200));
}

//-------------------
/// Returns byte-string data for specified document id from database collection.

void get_data_for_docid(std::string& sresp, const char* dbname, const char* colname, const char* docid, const char* urlws) {
  rapidjson::Document jdoc;
  get_doc_for_docid(jdoc, dbname, colname, docid, urlws);
  const char* dataid = jdoc["id_data"].GetString();
  MSG(DEBUG, "get_data_for_doc docid: " << docid << " -> dataid: " << dataid);
  get_data_for_id(sresp, dbname, dataid, urlws);
}

//-------------------
/// Returns byte-string data for specified document from database collection.

void get_data_for_doc(std::string& sresp, const char* dbname, const char* colname, const rapidjson::Value& jdoc, const char* urlws) {
  const char* docid = jdoc["_id"].GetString();
  MSG(DEBUG, "get_data_for_doc docid: \"" << docid << "\"");
  get_data_for_docid(sresp, dbname, colname, docid, urlws);
}

//-------------------
/// Returns database name with prefix, e.g. returns "cdb_cxi12345" for name="cxi12345".
  const std::string db_prefixed_name(const char* name, const char* prefix) {
    std::string s(prefix);
    s += name;
    MSG(DEBUG, "db_prefixed_name " << s << " size " << s.size());
    assert(s.size()< 128); 
    return s;
  }

//-------------------
/// Returns db_det, db_exp, det, query combined form input parameters.
/// e.g. const char* query = "{\"ctype\":\"pedestals\", \"run\":{\"$gte\": 87}}";
/// e.g. {"detector":"cspad_0001", "ctype":"pedestals", "run":{"$lte":10}, "time_sec":{"$lte":1234567890}}

//std::map<std::string, std::string>
//dbnames_collection_query(const char* det, const char* exp, const char* ctype, const unsigned run, const unsigned time_sec, const char* vers) {

void
dbnames_collection_query(std::map<std::string, std::string>& omap, const char* det, const char* exp, const char* ctype, const unsigned run, const unsigned time_sec, const char* vers) {
  MSG(DEBUG, "dbnames_collection_query for det: " << det);

  bool cond = (run>0) || (time_sec>0) || vers;
  assert(cond);

  const std::string db_det = (det) ? db_prefixed_name(det) : "";
  const std::string db_exp = (exp) ? db_prefixed_name(exp) : "";  

  omap["db_det"] = db_det;
  omap["db_exp"] = db_exp;
  omap["colname"] = (det) ? det : "";

  std::stringstream qss; 
  qss << "{|detector|:|" << det << '|'; 
  qss << ", |ctype|:|" << ctype << '|'; 
  if(run) qss << ", |run|:{|$lte|:" << run << '}'; 
  if(time_sec) qss << ", |time_sec|:{|$lte|:" << time_sec << '}';  
  if(vers) qss << ", |version|:|" << vers << '|';
  qss << '}';  
  std::string qs = qss.str();
  std::replace(qs.begin(), qs.end(), '|', '"');
  omap["query"] = qs;
  // MSG(DEBUG, "query  = " << qs);
}

//-------------------
/// Returns byte-string data and rapidjson::Document for specified parameters. 
/// Then byte-string data can be decoded to format defined in the document.

void calib_constants(std::string& sresp, rapidjson::Document& doc,
                     const char* det, const char* exp, const char* ctype, 
                     const unsigned run, const unsigned time_sec, const char* vers, const char* urlws) {
  MSG(DEBUG, "In calib_constants for det: " << det << " ctype: " << ctype);
 
  //assert(det);
  if(! det) {
    MSG(WARNING, "Collection/detector name is not defined");
    sresp.clear();
    return;
  }

  std::map<std::string, std::string> omap;
  dbnames_collection_query(omap, det, exp, ctype, run, time_sec, vers);

  const std::string& query   = omap["query"]; // "{\"ctype\":\"pedestals\", \"run\":{\"$lte\": 87}}";
  const std::string& db_det  = omap["db_det"];
  const std::string& db_exp  = omap["db_exp"];
  const std::string& colname = omap["colname"];

  MSG(DEBUG, "calib_constants:"
      << "\ncolname: " << colname
      << "\ndb_det : " << db_det
      << "\ndb_exp : " << db_exp
      << "\nquery  : " << query);

  // Use preferably experimental DB and the detector DB othervise
  const std::string& dbname = (exp) ? db_exp : db_det;

  rapidjson::Document outdocs;
  const rapidjson::Value& jdoc = find_doc(outdocs, dbname.c_str(), colname.c_str(), query.c_str(), urlws);

  //assert(! jdoc.IsNull());
  if(jdoc.IsNull()) {
    MSG(WARNING, "DOCUMENT IS NOT FOUND FOR QUERY = " << query);
    sresp.clear();
    return;
  }

  // Deep copy of found Value to output Document
  std::string sdoc = json_doc_to_string(jdoc);
  response_to_json_doc(sdoc, doc);
  //MSG(DEBUG, "doc : " << sdoc);

  get_data_for_doc(sresp, dbname.c_str(), colname.c_str(), doc, urlws);
}

//-------------------
// Returns NDArray of constants and Document with metadata

template<typename T> 
void calib_constants_nda(NDArray<T>& nda, rapidjson::Document& doc, const char* det, const char* exp, const char* ctype,
                         const unsigned run, const unsigned time_sec, const char* vers, const char* urlws) {
  MSG(TRACE, "In calib_constants_nda<T> for det: " << det << " ctype: " << ctype);

  std::string sresp;
  calib_constants(sresp, doc, det, exp, ctype, run, time_sec, vers, urlws);
  MSG(DEBUG, "doc: " << json_doc_to_string(doc));

  const std::string sshape = doc["data_shape"].GetString();
  nda.set_shape_string(sshape);
  nda.set_data_copy(sresp.data()); // deep copy string/byte array to nda data buffer

  MSG(DEBUG, "nda: " << nda);
} 

//-------------------
// Returns rapidjson::Document of constants and rapidjson::Document with metadata

void calib_constants_doc(rapidjson::Document& docdata, rapidjson::Document& doc, const char* det, const char* exp, const char* ctype,
                         const unsigned run, const unsigned time_sec, const char* vers, const char* urlws) {
  MSG(TRACE, "In calib_constants_doc for det: " << det << " ctype: " << ctype);

  std::string sresp;
  calib_constants(sresp, doc, det, exp, ctype, run, time_sec, vers, urlws);
  MSG(DEBUG, "calib_constants_doc:metadata doc: " << json_doc_to_string(doc));

  MSG(DEBUG, "calib_constants_doc:data: " << sresp);
  response_to_json_doc(sresp, docdata); // converts sresp -> rapidjson::Document
  MSG(DEBUG, "calib_constants_doc:data doc: " << json_doc_to_string(docdata));
} 

//-------------------

// string is used as a container for byte-string, but...
// PROBLEM: there is no way to pass external buffer to string witout copy,
// PROBLEM: string constructor always copies buffer in its own memory...

/*
template<typename T> 
void calib_constants_nda_unusable(NDArray<T>& nda, rapidjson::Document& doc, const char* det, const char* exp, const char* ctype,
                                  const unsigned run, const unsigned time_sec, const char* vers, const char* urlws) {
  MSG(TRACE, "In calib_constants_nda<T> for det: " << det << " ctype: " << ctype);

  //assert(det);
  if(! det) {
    MSG(WARNING, "Collection/detector name is not defined");
    nda.set_shape();
    return;
  }

  std::map<std::string, std::string> omap;
  dbnames_collection_query(omap, det, exp, ctype, run, time_sec, vers);

  const std::string& query   = omap["query"]; // "{\"ctype\":\"pedestals\", \"run\":{\"$lte\": 87}}";
  const std::string& db_det  = omap["db_det"];
  const std::string& db_exp  = omap["db_exp"];
  const std::string& colname = omap["colname"];

  MSG(DEBUG, "calib_constants:"
      << "\ncolname: " << colname
      << "\ndb_det : " << db_det
      << "\ndb_exp : " << db_exp
      << "\nquery  : " << query);

  // Use preferably experimental DB and the detector DB othervise
  const std::string& dbname = (exp) ? db_exp : db_det;

  rapidjson::Document outdocs;
  const rapidjson::Value& jdoc = find_doc(outdocs, dbname.c_str(), colname.c_str(), query.c_str(), urlws);

  //assert(! jdoc.IsNull());
  if(jdoc.IsNull()) {
    MSG(WARNING, "DOCUMENT IS NOT FOUND FOR QUERY = " << query);
    nda.set_shape();
    return;
  }

  // Deep copy of found Value to output Document
  std::string sdoc = json_doc_to_string(jdoc);
  response_to_json_doc(sdoc, doc);
  //MSG(DEBUG, "doc : " << sdoc);

  const std::string sshape = doc["data_shape"].GetString();
  nda.set_shape_string(sshape);
  size_t size = nda.size();
  nda.reserve_data_buffer(size);

  //=============================================================
  // s.data() buffer is not the same as nda.data() !!!!!!
  std::string s(reinterpret_cast<char const*>(nda.data()), size);
  //=============================================================

  get_data_for_doc(s, dbname.c_str(), colname.c_str(), doc, urlws);

  MSG(DEBUG, "nda: " << nda);
} 
*/

//-------------------
//#ifdef STRING01
//#undef STRING01
//#endif
#define STRING01(T)						\
  template void calib_constants_nda<T>(NDArray<T>&, rapidjson::Document&, const char*, const char*, const char*, const unsigned, const unsigned, const char*, const char*);
STRING01(int)
STRING01(float)
STRING01(double)

//template void calib_constants_nda<int>(NDArray<int>&, rapidjson::Document&, const char*, const char*, const char*, const unsigned, const unsigned, const char*, const char*);
//template void calib_constants_nda<float>(NDArray<float>&, rapidjson::Document&, const char*, const char*, const char*, const unsigned, const unsigned, const char*, const char*);
//template void calib_constants_nda<double>(NDArray<double>&, rapidjson::Document&, const char*, const char*, const char*, const unsigned, const unsigned, const char*, const char*);

//-------------------

#define STRING02(T) template void response_string_to_data_array<T>(const std::string&, const T*&, size_t&);
STRING02(int)
STRING02(float)
STRING02(double)

//template void response_string_to_data_array<int>   (const std::string&, const int*&,    size_t&); 
//template void response_string_to_data_array<float> (const std::string&, const float*&,  size_t&); 
//template void response_string_to_data_array<double>(const std::string&, const double*&, size_t&); 

//template class psalg::int_string_data_as_array<float>; 

//-------------------

} // namespace psalg

//-------------------

