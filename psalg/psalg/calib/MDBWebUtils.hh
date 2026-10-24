/**
 * @file
 * @brief Declares helpers for the calibration web service: libcurl HTTP requests, rapidjson parsing and printing, document lookup, and retrieval of calibration constants.
 */
#ifndef PSALG_MDBWEBUTILS_H
#define PSALG_MDBWEBUTILS_H

//-------------------------------------------
// Created on 2018-08-15 by Mikhail Dubrovin
//-------------------------------------------

/** Usage
 *
 *  #include "psalg/calib/MDBWebUtils.hh"
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

#include <map>
#include <string>
#include <vector>
//#include <sstream>
#include <stdint.h> // size_t;

#include <rapidjson/document.h>
#include "psalg/calib/NDArray.hh"

namespace psalg {

  /** Default base URL of the calibration web service, "https://pswww.slac.stanford.edu/calib_ws". */
  static const char *URLWS = "https://pswww.slac.stanford.edu/calib_ws";
  /** Null rapidjson::Value returned by find_doc() and value_from_json_doc() when nothing is found. Declared static, so each translation unit has its own copy. */
  static rapidjson::Value EMPTY_VALUE;

//-------------------

  /** Print "In print_json_doc_as_string: " followed by json_doc_to_string(value) to stdout. */
  void print_json_doc_as_string(const rapidjson::Value& value);
  /** Recursively print doc to stdout with a type label per value, indenting nested levels by 4 spaces plus offset; strings are cut to 1000 characters. */
  void print_json_doc(const rapidjson::Value& doc, const std::string& offset="  ");
  /** Print "In print_vector_of_strings" and then each string of v prefixed by gap, one per line, to stdout. */
  void print_vector_of_strings(const std::vector<std::string>& v, const char* gap="   ==  ");
  /** Reinterpret the bytes of s as floats and print their count and the first nvals values to stdout. nvals is not limited to the data size. */
  void print_byte_string_as_float(const std::string& s, const size_t nvals=100);

  /** Return a type name for v: "string", "int" (also used for booleans), "number", "object", "array", or a "Non-implemented type: " string for other types (e.g. null). */
  std::string json_value_type(const rapidjson::Value& v);
  /** Serialize d to a compact JSON string with rapidjson::Writer. */
  std::string json_doc_to_string(const rapidjson::Value& d);
  /** Parse the JSON text s into jd; on a parse error log an ERROR message with the error code and offset. */
  void chars_to_json_doc(const char* s, rapidjson::Document& jd);
  /** Declared here; no definition was found in MDBWebUtils.cc. */
  rapidjson::Document chars_to_json_doc(const char* s);
  /** Clear vout and, if doc is a non-empty array, fill it with the array's string elements. */
  void json_doc_to_vector_of_strings(const rapidjson::Value& doc, std::vector<std::string>& vout);
  /** libcurl write callback: append size*nmemb bytes from buf to the std::string pointed to by up and return that byte count. */
  size_t _callback(char* buf, size_t size, size_t nmemb, void* up);
  /** Parse sresp as JSON into the given document (chars_to_json_doc()). */
  void response_to_json_doc(const std::string& sresp, rapidjson::Document&);
  /** Set surl to urlws/dbname/colname, followed by "?query_string=" and the URL-escaped query if query is not NULL. */
  void string_url_with_query(std::string& surl, const char* dbname, const char* colname, const char* query=NULL, const char* urlws=URLWS);

  /**
   * HTTP GET url with libcurl and store the response body in sresp (cleared first).
   * Logs a WARNING if no content type is received; errors from curl_easy_perform() are not reported.
   */
  void request(std::string& sresp, const char* url=URLWS);
  /** GET urlws and fill dbnames with the string elements of the returned JSON array (left empty if it is not a non-empty array). */
  void database_names(std::vector<std::string>& dbnames, const char* urlws=URLWS);
  /** GET urlws/dbname and fill colnames with the string elements of the returned JSON array (left empty if it is not a non-empty array). */
  void collection_names(std::vector<std::string>& colnames, const char* dbname, const char* urlws=URLWS);
  /** GET the URL built by string_url_with_query() and store the raw response in sresp. */
  void find_docs(std::string& sresp, const char* dbname, const char* colname, const char* query=NULL, const char* urlws=URLWS);
  /**
   * Run find_docs(), parse the response into outdocs, and return the element with the largest "time_sec" if query contains "time_sec", else the largest "run".
   * Returns EMPTY_VALUE with a WARNING if the response is not an array or is empty; query must not be NULL. The returned reference points into outdocs.
   */
  const rapidjson::Value& find_doc(rapidjson::Document& outdocs, const char* dbname, const char* colname, const char* query=NULL, const char* urlws=URLWS);
  /** Return doc[valname], or EMPTY_VALUE if doc has no such member. */
  const rapidjson::Value& value_from_json_doc(const rapidjson::Value& doc, const char* valname);
  /** GET urlws/dbname/colname/docid and parse the response into jdoc. */
  void get_doc_for_docid(rapidjson::Document& jdoc, const char* dbname, const char* colname, const char* docid, const char* urlws=URLWS);
  /** GET urlws/dbname/gridfs/dataid and store the raw response bytes in sresp. */
  void get_data_for_id(std::string& sresp, const char* dbname, const char* dataid, const char* urlws=URLWS);
  /** Fetch document docid (get_doc_for_docid()), then fetch into sresp the data whose id is the document's "id_data" string (get_data_for_id()). */
  void get_data_for_docid(std::string& sresp, const char* dbname, const char* colname, const char* docid, const char* urlws=URLWS);
  /** Call get_data_for_docid() with the "_id" string member of jdoc. */
  void get_data_for_doc(std::string& sresp, const char* dbname, const char* colname, const rapidjson::Value& jdoc, const char* urlws=URLWS);


  /** Return prefix + name (e.g. "cdb_" + name); asserts that the result is shorter than 128 characters. */
  const std::string db_prefixed_name(const char* name, const char* prefix="cdb_");

//void
//  std::map<std::string, std::string>
//  dbnames_collection_query(const char* det, const char* exp=NULL, const char* ctype=NULL, const unsigned run=0, const unsigned time_sec=0, const char* version=NULL);
  /**
   * Fill omap with "db_det" ("cdb_" + det), "db_exp" ("cdb_" + exp, or empty if exp is NULL), "colname" (det) and "query".
   * The query is a JSON string with "detector" and "ctype", plus "run":{"$lte":run} if run > 0, "time_sec":{"$lte":time_sec} if time_sec > 0 and "version" if version is not NULL.
   * Asserts that run > 0, time_sec > 0 or version is not NULL.
   */
  void
  dbnames_collection_query(std::map<std::string, std::string>& omap, const char* det, const char* exp=NULL, const char* ctype=NULL, const unsigned run=0, const unsigned time_sec=0, const char* version=NULL);

  /**
   * Find the best-matching document (find_doc()) in collection detector of database "cdb_" + experiment if experiment is given, else "cdb_" + detector, and deep-copy it into doc.
   * Logs a WARNING and leaves doc unchanged if detector is NULL or no document is found.
   */
  void calib_doc(rapidjson::Document& doc, const char* detector, const char* experiment=NULL, const char* ctype=NULL, const unsigned run=0, const unsigned time_sec=0, const char* version=NULL, const char* urlws=URLWS);

  /**
   * Like calib_doc(), then download the data referenced by the found document into sresp (get_data_for_doc()).
   * If detector is NULL or no document is found, logs a WARNING and clears sresp.
   */
  void calib_constants(std::string& sresp, rapidjson::Document& doc, const char* detector, const char* experiment=NULL, const char* ctype=NULL, const unsigned run=0, const unsigned time_sec=0, const char* version=NULL, const char* urlws=URLWS);


/**
 * Run calib_constants(), set the shape of nda from the document's "data_shape" string, and deep-copy the downloaded bytes into nda.
 * The document is assumed to have "data_shape" (not checked). Instantiated in MDBWebUtils.cc for int, float, double, uint16_t and uint32_t.
 */
template<typename T> 
void calib_constants_nda(NDArray<T>& nda, rapidjson::Document& doc, const char* detector, const char* experiment=NULL, const char* ctype=NULL, const unsigned run=0, const unsigned time_sec=0, const char* version=NULL, const char* urlws=URLWS);


/** Run calib_constants() with docmeta as the metadata document, then parse the downloaded data as JSON into docdata. */
void calib_constants_doc(rapidjson::Document& docdata, rapidjson::Document& docmeta, const char* detector, const char* experiment=NULL, const char* ctype=NULL, const unsigned run=0, const unsigned time_sec=0, const char* version=NULL, const char* urlws=URLWS);

//-------------------

/**
 * Point pout at the bytes of s reinterpreted as TDATA and set size to s.size()/sizeof(TDATA). Nothing is copied, so pout is valid only while s is unchanged.
 * Instantiated in MDBWebUtils.cc for int, float, double, uint16_t and uint32_t.
 */
template<typename TDATA> void response_string_to_data_array(const std::string& s, const TDATA*& pout, size_t& size);

//-------------------

} // namespace psalg

#endif // PSALG_MDBWEBUTILS_H
