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
 *   _ = wu.database_names(url=cc.URL)
 *   _ = wu.collection_names(dbname, url=cc.URL)
 *   _ = wu.find_docs(dbname, colname, query={'ctype':'pedestals'}, url=cc.URL)
 *   _ = wu.find_doc(dbname, colname, query={'ctype':'pedestals'}, url=cc.URL)
 *   _ = wu.get_doc_for_docid(dbname, colname, docid, url=cc.URL)
 *   _ = wu.get_data_for_id(dbname, dataid, url=cc.URL)
 *   _ = wu.get_data_for_docid(dbname, colname, docid, url=cc.URL)
 *   _ = wu.get_data_for_doc(dbname, colname, doc, url=cc.URL)
 *   _ = wu.calib_constants(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, url=cc.URL)
 *
 */

#include <string>
#include <vector>
//#include <sstream>
#include <stdint.h> // size_t;

#include <rapidjson/document.h>


namespace psalg {

  static const char *URL = "https://pswww-dev.slac.stanford.edu/calib_ws";

//-------------------

  void print_json_value(const rapidjson::Value &value);
  void print_json_doc(const rapidjson::Value &doc, const std::string& offset=" ");
  void print_vector_of_strings(const std::vector<std::string>& v);

  std::string json_doc_to_string(const rapidjson::Document &doc);
  rapidjson::Document chars_to_json_doc(const char* s);
  const std::vector<std::string>& json_doc_to_vector_of_strings(const rapidjson::Document &doc);

  size_t string_callback(char* buf, size_t size, size_t nmemb, void* up);
  size_t json_callback(char* buf, size_t size, size_t nmemb, void* up);

  const std::string& string_response();
  const rapidjson::Document & json_doc_response();

  void request(const char* url=URL, const char* query="");
  const std::vector<std::string>& database_names(const char* url=URL);
  const std::vector<std::string>& collection_names(const char* dbname, const char* url=URL);

//-------------------

} // namespace psalg

#endif // PSALG_MDBWEBUTILS_H
