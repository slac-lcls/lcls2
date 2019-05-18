#ifndef PSALG_CALIBPARSDBWEB_H
#define PSALG_CALIBPARSDBWEB_H
//-----------------------------

#include "psalg/calib/CalibParsDB.hh" // #include "psalg/calib/Query.hh"
#include "psalg/calib/MDBWebUtils.hh"

using namespace psalg; // for NDArray

namespace calib {

//-----------------------------

class CalibParsDBWeb : public CalibParsDB {
public:

  CalibParsDBWeb();
  virtual ~CalibParsDBWeb();

  virtual const NDArray<float>&    get_ndarray_float (Query&);
  virtual const NDArray<double>&   get_ndarray_double(Query&);
  virtual const NDArray<uint16_t>& get_ndarray_uint16(Query&);
  virtual const NDArray<uint32_t>& get_ndarray_uint32(Query&);
  virtual const std::string&       get_string        (Query&);

  //virtual const ResponseDB& get_responce(Query&);
 
  CalibParsDBWeb(const CalibParsDBWeb&) = delete;
  CalibParsDBWeb& operator = (const CalibParsDBWeb&) = delete;

private:

  void _default_msg(const std::string& msg=std::string()) const;

}; // class

//-----------------------------

} // namespace calib

#endif // PSALG_CALIBPARSDBWEB_H
