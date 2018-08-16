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

//#include <iostream> //ostream
#include <string>
#include <vector>
#include <sstream>

namespace psalg {

  static const char *URL = "https://pswww-dev.slac.stanford.edu/calib_ws";

//-------------------

  const std::string request_url_query(const char* url=URL, const char* query="");
  const std::vector<char*> database_names(const char* url=URL);
  const std::vector<char*> collection_names(const char* dbname, const char* url=URL);

//-------------------

} // namespace psalg

#endif // PSALG_MDBWEBUTILS_H
