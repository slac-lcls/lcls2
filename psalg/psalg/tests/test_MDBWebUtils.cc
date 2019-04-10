/* 
 * https://curl.haxx.se/libcurl/c/getinfo.html
 */

#include "psalg/utils/Logger.hh" // MSG, LOGGER,...

#include <stdio.h>
#include <iostream>
#include <sstream>

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
// returns "https://pswww-dev...
// returns "https://pswww.slac.stanford.edu/calib_ws/cdb_cspad_0001/cspad_0001?query_string=%7B%22ctype%22%3A+%22pedestals%22%7D"
// curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_cspad_0001/cspad_0001?query_string=%7B%22ctype%22%3A+%22pedestals%22%7D"
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
 
// curl -s "https://pswww-dev.slac.stanford.edu/calib_ws/cdb_cspad_0001/cspad_0001?query_string=..."
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
  const char* dbname1 = "cdb_cxid9114";
  const char* dataid1 = "5b6cdde71ead144f11531974"; // for PEDESTALS
  std::cout << "In test_get_data_for_id PEDESTALS for " << dbname1 << " and " << dataid1 << ":\n";  
  std::string sresp1;
  get_data_for_id(sresp1, dbname1, dataid1);
  print_byte_string_as_float(sresp1, 100);

  print_hline(80,'_');
  const char* dbname2 = "cdb_cspad_0001";
  const char* dataid2 = "5b6cdde71ead144f11531999"; // for PEDESTALS
  std::cout << "In test_get_data_for_id PEDESTALS for " << dbname2 << " and " << dataid2 << ":\n";  
  std::string sresp2;
  get_data_for_id(sresp2, dbname2, dataid2);
  print_byte_string_as_float(sresp2, 100);

  print_hline(80,'_');
  const char* dbname3 = "cdb_cspad_0001";
  const char* dataid3 = "5b6cdef31ead1451fc5f0a2e"; // for GEOMETRY
  std::cout << "In test_get_data_for_id GEOMETRY for " << dbname3 << " and " << dataid3 << ":\n";  
  std::string sresp3;
  get_data_for_id(sresp3, dbname3, dataid3);
  std::cout << sresp3 << '\n';
}

//-------------------

void test_get_data_for_docid() {
  const char* dbname  = "cdb_cspad_0001";
  const char* colname = "cspad_0001"; 
  const char* docid = "5b6cdde71ead144f115319be"; // docid for PEDESTALS
  std::cout << "In test_get_data_for_docid PEDESTALS for " << dbname << " and " << colname << ":\n";  
  std::string sresp;
  get_data_for_docid(sresp, dbname, colname, docid);
  //std::cout << sresp << '\n';
  print_byte_string_as_float(sresp, 20);
}

//-------------------

void test_get_data_for_doc() {
  printf("In test_get_data_for_doc\n");  
  const char* dbname  = "cdb_cspad_0001";
  const char* colname = "cspad_0001"; 
  //const char* query   =  "{\"ctype\":\"pedestals\", \"run\":{\"$gte\": 87}}"; // 2 db docs
  //const char* query   =  "{\"ctype\":\"pedestals\", \"run\":78}";           // 1 db doc
  //const char* query   =  "{\"ctype\":\"pedestals\", \"run\":1234}";         // 0 db docs
  //const char* query   =  "{\"ctype\":\"pedestals\"}";                       // many db docs
  const char* query   =  "{\"ctype\":\"geometry\"}";                       // many db docs
  rapidjson::Document jdoc;
  const rapidjson::Value& jvalue = find_doc(jdoc, dbname, colname, query);
  std::string sresp;
  get_data_for_doc(sresp, dbname, colname, jvalue);
  std::cout << sresp << '\n';
}

//-------------------

void test_db_prefixed_name() {
  printf("In test_db_prefixed_name\n");  
  const char* name = "cspad_0001";
  const std::string s = db_prefixed_name(name);
  std::cout << "db_prefixed_name:" << s << '\n';
}

//-------------------

void test_dbnames_collection_query() {
  printf("In test_dbnames_collection_query\n");  
  const char* det = "cspad_0001"; 
  const char* exp = "cxid9114"; 
  const char* ctype = "pedestals";
  const unsigned run = 120;
  const unsigned time_sec = 1534567890;
  const char* vers = NULL;
  std::map<std::string, std::string> omap;
  dbnames_collection_query(omap, det, exp, ctype, run, time_sec, vers);

  std::cout << "  db_exp = " << omap["db_exp"]
  	    << "\ndb_det = " << omap["db_det"]
  	    << "\ncolname= " << omap["colname"]
  	    << "\nquery  = " << omap["query"]
  	    << '\n';
}

//-------------------

void test_calib_constants() {
  printf("In test_calib_constants\n");  
  const char* det = "cspad_0001"; 
  const char* exp = "cxid9114"; 
  const char* ctype = "pedestals";
  const unsigned run = 150;
  const unsigned time_sec = 1534567890;
  const char* vers = NULL;
  std::string sresp;
  rapidjson::Document doc;
  std::cout << "exp:" << exp << " run:" << run  << " det:" <<  det << '\n';
  calib_constants(sresp, doc, det, exp, ctype, run, time_sec, vers);
  std::cout << "doc : " << json_doc_to_string(doc) << '\n';

  //std::cout << "TODO convert and show \"sresp\"\n";
  print_byte_string_as_float(sresp, 20);
}

//-------------------

void test_calib_constants_nda() {
  printf("In test_calib_constants_nda\n");  
  const char* det = "cspad_0001"; 
  const char* exp = "cxid9114"; 
  const char* ctype = "pedestals";
  const unsigned run = 150;
  const unsigned time_sec = 1534567890;
  const char* vers = NULL;
  typedef float pedestals_t;
  NDArray<pedestals_t> nda;
  rapidjson::Document doc;

  MSG(INFO, "XXXX: Begin calib_constants_nda");
  calib_constants_nda<pedestals_t>(nda, doc, det, exp, ctype, run, time_sec, vers);
  MSG(INFO, "XXXX: Done");

  std::cout << "doc: " << json_doc_to_string(doc)
	    << "\nSome matadata from doc:" 
	    << " shape: " << doc["data_shape"].GetString()
	    << " type: "  << doc["data_type"] .GetString()
	    << " dtype: " << doc["data_dtype"].GetString()
	    << " size: "  << doc["data_size"] .GetString()
	    << " ndim: "  << doc["data_ndim"] .GetString()
	    << "\nnda: "  << nda << '\n';
}

//-------------------

void test_calib_constants_doc() {
  printf("In test_calib_constants_doc\n");  
  const char* det = "opal1000_0059";  // XrayTransportDiagnostic.0:Opal1000.0
  const char* exp = "amox23616";    // amox23616
  const char* ctype = "lasingoffreference"; // lasingoffreference
  const unsigned run = 60;        // 56 
  const unsigned time_sec = 1534567890;
  const char* vers = NULL;
  rapidjson::Document docdata;
  rapidjson::Document docmeta;

  MSG(INFO, "== Begin calib_constants_doc");
  calib_constants_doc(docdata, docmeta, det, exp, ctype, run, time_sec, vers);
  MSG(INFO, "== Done");

  std::cout << "== docmeta: " << json_doc_to_string(docmeta) << '\n';
  std::cout << "== docdata: " << json_doc_to_string(docdata) << '\n';
  //    det = 'opal1000_0059'
  //    #data, doc = calib_constants(det, exp='amox23616', ctype='lasingoffreference', run=60, time_sec=None, vers=None)
  //    data, doc = calib_constants(det, exp=None, ctype='lasingoffreference', run=60, time_sec=None, vers=None)
 }

//-------------------

std::string usage(const std::string& tname="")
{
  std::stringstream ss;
  if (tname == "") ss << "Usage command> test_MDBWebUtils <test-number>\n  where test-number";
  if (tname == "" || tname=="0"	) ss << "\n  0  - test_request()";
  if (tname == "" || tname=="1"	) ss << "\n  1  - test_database_names()";
  if (tname == "" || tname=="2"	) ss << "\n  2  - test_collection_names()";
  if (tname == "" || tname=="3"	) ss << "\n  3  - std::string url = test_string_url_with_query()";
  if (tname == "" || tname=="4"	) ss << "\n  4  - test_request_with_query()";
  if (tname == "" || tname=="5"	) ss << "\n  5  - test_find_docs()";
  if (tname == "" || tname=="6"	) ss << "\n  6  - test_find_doc()";
  if (tname == "" || tname=="7"	) ss << "\n  7  - test_get_doc_for_docid()";
  if (tname == "" || tname=="8"	) ss << "\n  8  - test_get_data_for_id()";
  if (tname == "" || tname=="9"	) ss << "\n  9  - test_get_data_for_docid()";
  if (tname == "" || tname=="10") ss << "\n  10 - test_get_data_for_doc()";
  if (tname == "" || tname=="11") ss << "\n  11 - test_db_prefixed_name()";
  if (tname == "" || tname=="12") ss << "\n  12 - test_dbnames_collection_query()";
  if (tname == "" || tname=="13") ss << "\n  13 - test_calib_constants()";
  if (tname == "" || tname=="14") ss << "\n  14 - test_calib_constants_nda()";
  if (tname == "" || tname=="15") ss << "\n  15 - test_calib_constants_doc()";
  ss << '\n';
  return ss.str();
}

//-------------------

int main(int argc, char* argv[])
{
  print_hline(80,'_');
  //MSG(INFO, LOGGER.tstampStart() << " Logger started"); // optional record
  LOGGER.setLogger(LL::DEBUG, "%H:%M:%S.%f");           // set level and time format
  MSG(INFO, "In test_MDBWebUtils");

  cout << usage(); 
  print_hline(80,'_');
  if (argc==1) {return 0;}
  std::string tname(argv[1]);
  cout << usage(tname); 

  if      (tname=="0")  test_request();
  else if (tname=="1")  test_database_names();
  else if (tname=="2")  test_collection_names();
  else if (tname=="3")  std::string url = test_string_url_with_query();
  else if (tname=="4")  test_request_with_query();
  else if (tname=="5")  test_find_docs();
  else if (tname=="6")  test_find_doc();
  else if (tname=="7")  test_get_doc_for_docid();
  else if (tname=="8")  test_get_data_for_id();
  else if (tname=="9")  test_get_data_for_docid();
  else if (tname=="10") test_get_data_for_doc();
  else if (tname=="11") test_db_prefixed_name();
  else if (tname=="12") test_dbnames_collection_query();
  else if (tname=="13") test_calib_constants();
  else if (tname=="14") test_calib_constants_nda();
  else if (tname=="15") test_calib_constants_doc();
  else MSG(WARNING, "Undefined test name \"" << tname << '\"');
 
  print_hline(80,'_');
  return EXIT_SUCCESS;
}

//-------------------

