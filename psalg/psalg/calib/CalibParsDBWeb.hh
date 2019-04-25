#ifndef PSALG_CALIBPARSDBWEB_H
#define PSALG_CALIBPARSDBWEB_H
//-----------------------------

#include "psalg/calib/CalibParsDB.hh"

using namespace psalg; // for NDArray

namespace calib {

//-----------------------------

class CalibParsDBWeb : public CalibParsDB {
public:

  CalibParsDBWeb();
  virtual ~CalibParsDBWeb();

  virtual const NDArray<double>&   get_ndarray_double(const Query&);
  virtual const NDArray<float>&    get_ndarray_float (const Query&);
  virtual const NDArray<uint16_t>& get_ndarray_uint16(const Query&);
  virtual const NDArray<uint32_t>& get_ndarray_uint32(const Query&);
  //virtual const std::string&       get_string        (const Query&);

  CalibParsDBWeb(const CalibParsDBWeb&) = delete;
  CalibParsDBWeb& operator = (const CalibParsDBWeb&) = delete;

private:

  void _default_msg(const std::string& msg=std::string()) const;

}; // class

//-----------------------------

} // namespace calib

#endif // PSALG_CALIBPARSDBWEB_H
