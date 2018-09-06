#ifndef PSALG_MDBWEBUTILS_H
#define PSALG_MDBWEBUTILS_H

//-------------------------------------------
// Created on 2018-08-15 by Mikhail Dubrovin
//-------------------------------------------

/** Usage
 *
 *  #include "psalg/calib/MDBWebUtils.hh"
 *
 *
 *
 *
 *
 *   _ = wu.requests_get(url, query=None)
 *   _ = wu.database_names(urlws=cc.URLWS)
 *   _ = wu.collection_names(dbname, urlws=cc.URLWS)
 *   _ = wu.find_docs(dbname, colname, query={'ctype':'pedestals'}, urlws=cc.URLWS)
 *   _ = wu.find_doc(dbname, colname, query={'ctype':'pedestals'}, urlws=cc.URLWS)
 *   _ = wu.get_doc_for_docid(dbname, colname, docid, urlws=cc.URLWS)
 *   _ = wu.get_data_for_id(dbname, dataid, urlws=cc.URLWS)
 *   _ = wu.get_data_for_docid(dbname, colname, docid, urlws=cc.URLWS)
 *   _ = wu.get_data_for_doc(dbname, colname, doc, urlws=cc.URLWS)
 *   _ = wu.calib_constants(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, urlws=cc.URLWS)
 *
 */

#include <string>
#include <vector>
//#include <sstream>
#include <stdint.h> // size_t;

#include <rapidjson/document.h>


namespace psalg {

  static const char *URLWS = "https://pswww-dev.slac.stanford.edu/calib_ws";
  static rapidjson::Value EMPTY_VALUE;

//-------------------

  void print_json_doc_as_string(const rapidjson::Value& value);
  void print_json_doc(const rapidjson::Value& doc, const std::string& offset="  ");
  void print_vector_of_strings(const std::vector<std::string>& v, const char* gap="   ==  ");
  void print_byte_string_as_float(const std::string& s, const size_t nvals=100);

  std::string json_value_type(const rapidjson::Value& v);
  std::string json_doc_to_string(const rapidjson::Value& d);
  void chars_to_json_doc(const char* s, rapidjson::Document& jd);
  rapidjson::Document chars_to_json_doc(const char* s);
  void json_doc_to_vector_of_strings(const rapidjson::Value& doc, std::vector<std::string>& vout);
  size_t _callback(char* buf, size_t size, size_t nmemb, void* up);
  void response_to_json_doc(const std::string& sresp, rapidjson::Document&);
  void string_url_with_query(std::string& surl, const char* dbname, const char* colname, const char* query=NULL, const char* urlws=URLWS);

  void request(std::string& sresp, const char* url=URLWS);
  void database_names(std::vector<std::string>& dbnames, const char* urlws=URLWS);
  void collection_names(std::vector<std::string>& colnames, const char* dbname, const char* urlws=URLWS);
  void find_docs(std::string& sresp, const char* dbname, const char* colname, const char* query=NULL, const char* urlws=URLWS);
  const rapidjson::Value& find_doc(rapidjson::Document& outdocs, const char* dbname, const char* colname, const char* query=NULL, const char* urlws=URLWS);
  const rapidjson::Value& value_from_json_doc(const rapidjson::Value& doc, const char* valname);
  void get_doc_for_docid(rapidjson::Document& jdoc, const char* dbname, const char* colname, const char* docid, const char* urlws=URLWS);
  void get_data_for_id(std::string& sresp, const char* dbname, const char* dataid, const char* urlws=URLWS);
  void get_data_for_docid(std::string& sresp, const char* dbname, const char* colname, const char* docid, const char* urlws=URLWS);
  void get_data_for_doc(std::string& sresp, const char* dbname, const char* colname, const rapidjson::Value& jdoc, const char* urlws=URLWS);

  void calib_constants(std::string& sresp, const char* detector, const char* experiment=NULL, const char* ctype=NULL, const unsigned run=0, const unsigned time_sec=0, const char* version=NULL, const char* urlws=URLWS);

//-------------------

template<typename TDATA> void response_string_to_data_array(const std::string& s, const TDATA*& pout, size_t& size);

//-------------------

} // namespace psalg

#endif // PSALG_MDBWEBUTILS_H
