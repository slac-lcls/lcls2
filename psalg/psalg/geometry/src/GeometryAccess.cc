//-------------------

#include <iostream> // for cout
#include <fstream>  // for ifstream 
#include <sstream>  // for stringstream
#include <iomanip>  // for setw, setfill

#include "psalg/geometry/GeometryAccess.hh"
#include "psalg/geometry/GeometryObject.hh"
#include "psalg/utils/Logger.hh" // MSG, LOGGER

//-------------------

namespace psalg {

GeometryAccess::GeometryAccess (const std::string& path, unsigned pbits)
  : m_path(path)
  , m_pbits(pbits)
  , p_iX(0)
  , p_iY(0)
  , p_image(0)
  , p_XatZ(0)
  , p_YatZ(0)
{
  if(m_pbits & 1) MSG(INFO, "m_pbits = " << m_pbits); 

  load_pars_from_file();

  if(m_pbits & 2) print_list_of_geos();
  if(m_pbits & 4) print_list_of_geos_children();
  if(m_pbits & 8) print_comments_from_dict();
}

//--------------

GeometryAccess::~GeometryAccess ()
{
  delete [] p_iX;
  delete [] p_iY;
  delete p_image;
  delete [] p_XatZ;
  delete [] p_YatZ;
}

//-------------------
void GeometryAccess::load_pars_from_file(const std::string& path)
{
  m_dict_of_comments.clear();
  v_list_of_geos.clear();
  v_list_of_geos.reserve(100);

  if(! path.empty()) m_path = path;

  std::ifstream f(m_path.c_str());

  if(not f.good()) { MSG(ERROR, "Calibration file " << m_path << " does not exist"); }
  else             { if(m_pbits & 1) MSG(INFO, "load_pars_from_file(): " << m_path); }

  std::string line;
  while (std::getline(f, line)) {    
    if(m_pbits & 256) std::cout << line << '\n'; // << " length = " << line.size() << '\n';
    if(line.empty())    continue;    // discard empty lines
    if(line[0] == '#') {          // process line of comments 
       add_comment_to_dict(line); 
       continue;
    }
    // make geometry object and add it in the list
    v_list_of_geos.push_back( parse_line(line) );    
  }

  f.close();

  set_relations();
}

//-------------------

void GeometryAccess::save_pars_in_file(const std::string& path)
{
  if(m_pbits & 1) MSG(INFO, "Save pars in file " << path.c_str());

  std::stringstream ss;

  // Save comments
  std::map<int, std::string>::iterator iter;
  for (iter = m_dict_of_comments.begin(); iter != m_dict_of_comments.end(); ++iter) {
    //ss << "# " << std::setw(10) << std::left << iter->first;
    ss << "# " << iter->second << '\n';
  }

  ss << '\n';

  // Save data
  for(std::vector<GeometryAccess::shpGO>::iterator it  = v_list_of_geos.begin(); 
                                                   it != v_list_of_geos.end(); ++it) {
    if( (*it)->get_parent_name().empty() ) continue;
    ss << (*it)->str_data() << '\n';
  }

  std::ofstream out(path.c_str()); 
  out << ss.str();
  out.close();

  if(m_pbits & 64) std::cout << ss.str();
}

//-------------------

void GeometryAccess::add_comment_to_dict(const std::string& line)
{ 
  std::size_t p1 = line.find_first_not_of("#");
  std::size_t p2 = line.find_first_not_of(" ", p1);
  //std::size_t p2 = line.find_first_of(" ", p1);

  //std::cout << "  p1:" << p1 << "  p2:" << p2 << '\n';
  //if (p1 == std::string::npos) ...
  //std::cout << "comment: " << line << '\n'; 
  //std::cout << "   split line: [" << beginline << "] = " << endline << '\n'; 

  if (p1 == std::string::npos) return; // available # but missing keyword

  int ind = m_dict_of_comments.size();

  if (p2 == std::string::npos) { // keyword is available but comment is missing
    m_dict_of_comments[ind] = std::string();
    return;
  }

  std::string endline(line, p2);
  m_dict_of_comments[ind] = endline;
}

//-------------------

/*
void GeometryAccess::add_comment_to_dict(const std::string& line)
{ 
  std::size_t p1 = line.find_first_not_of("# ");
  std::size_t p2 = line.find_first_of(" ", p1);
  std::size_t p3 = line.find_first_not_of(" ", p2);

  //std::cout << "  p1:" << p1 << "  p2:" << p2 << "  p3:" << p3 << '\n';
  //if (p1 == std::string::npos) ...
  //std::cout << "comment: " << line << '\n'; 
  //std::cout << "   split line: [" << beginline << "] = " << endline << '\n'; 

  if (p1 == std::string::npos) return; // available # but missing keyword
  if (p2 == std::string::npos) return;

  std::string beginline(line, p1, p2-p1);

  if (p3 == std::string::npos) { // keyword is available but comment is missing
    m_dict_of_comments[beginline] = std::string();
    return;
  }

  std::string endline(line, p3);
  m_dict_of_comments[beginline] = endline;
}
*/

//-------------------

GeometryAccess::shpGO GeometryAccess::parse_line(const std::string& line)
{
  std::string pname;
  unsigned    pindex;
  std::string oname;
  unsigned    oindex;
  double      x0;
  double      y0;
  double      z0;
  double      rot_z;
  double      rot_y;
  double      rot_x;                  
  double      tilt_z;
  double      tilt_y;
  double      tilt_x; 

  std::stringstream ss(line);

  if(ss >> pname >> pindex >> oname >> oindex >> x0 >> y0 >> z0 
        >> rot_z >> rot_y >> rot_x >> tilt_z >> tilt_y >> tilt_x) {
      GeometryAccess::shpGO shp( new GeometryObject (pname,
                              		     pindex,
                              		     oname,
                              		     oindex,
                              		     x0,
                              		     y0,
                              	  	     z0,
                              		     rot_z,
                              		     rot_y,
                              		     rot_x,                  
                              		     tilt_z,
                              		     tilt_y,
                              		     tilt_x 
		                             ));
      if(m_pbits & 256) shp->print_geo();
      return shp;
  }
  else {
      std::string msg = "parse_line(...) can't parse line: " + line;
      //std::cout << msg;
      MSG(ERROR, msg);
      return GeometryAccess::shpGO();
  }
}

//-------------------

GeometryAccess::shpGO GeometryAccess::find_parent(const GeometryAccess::shpGO& geobj)
{
  for(std::vector<GeometryAccess::shpGO>::iterator it  = v_list_of_geos.begin(); 
                                   it != v_list_of_geos.end(); ++it) {
    if(*it == geobj) continue; // skip geobj themself
    if(   (*it)->get_geo_index() == geobj->get_parent_index()
       && (*it)->get_geo_name()  == geobj->get_parent_name() ) {
      return (*it);
    }
  }

  //The name of parent object is not found among geos in the v_list_of_geos
  // add top parent object to the list

  if(m_pbits & 256) std::cout << "  GeometryAccess::find_parent(...): parent is not found..."
                              << geobj->get_parent_name() << " idx:" << geobj->get_parent_index() << '\n';

  if( ! geobj->get_parent_name().empty() ) { // skip top parent itself

    if(m_pbits & 256) std::cout << "  create one with name:" << geobj->get_parent_name() 
                                << " idx:" << geobj->get_parent_index() << '\n';

    GeometryAccess::shpGO shp_top_parent( new GeometryObject (std::string(),
                            		                      0,
                            		                      geobj->get_parent_name(),
                            		                      geobj->get_parent_index()));

    v_list_of_geos.push_back( shp_top_parent );
    return shp_top_parent;		  
  }

  if(m_pbits & 256) std::cout << "  return empty shpGO() for the very top parent\n";
  return GeometryAccess::shpGO(); // for top parent itself
}

//-------------------

void GeometryAccess::set_relations()
{
  if(m_pbits & 16) MSG(INFO, "Begin set_relations(): size of the list:" << v_list_of_geos.size());
  for(std::vector<GeometryAccess::shpGO>::iterator it  = v_list_of_geos.begin(); 
                                                   it != v_list_of_geos.end(); ++it) {

    GeometryAccess::shpGO shp_parent = find_parent(*it);
    //std::cout << "set_relations(): found parent name:" << shp_parent->get_parent_name()<<'\n';

    if( shp_parent == GeometryAccess::shpGO() ) continue; // skip parent of the top object
    
    (*it)->set_parent(shp_parent);
    shp_parent->add_child(*it);

    if(m_pbits & 16) 
      std::cout << "\n  geo:"     << std::setw(10) << (*it) -> get_geo_name()
                << " : "                           << (*it) -> get_geo_index()
                << " has parent:" << std::setw(10) << shp_parent -> get_geo_name()
                << " : "                           << shp_parent -> get_geo_index()
                << '\n';
  }
}

//-------------------

GeometryAccess::shpGO GeometryAccess::get_geo(const std::string& oname, const unsigned& oindex)
{
  for(std::vector<GeometryAccess::shpGO>::iterator it  = v_list_of_geos.begin(); 
                                   it != v_list_of_geos.end(); ++it) {
    if(   (*it)->get_geo_index() == oindex
       && (*it)->get_geo_name()  == oname ) 
          return (*it);
  }
  return GeometryAccess::shpGO(); // None
}

//-------------------

GeometryAccess::shpGO GeometryAccess::get_top_geo()
{
  return v_list_of_geos.back();
}

//-------------------

void
GeometryAccess::get_pixel_coords(const double*& X, 
                                 const double*& Y, 
                                 const double*& Z, 
				 unsigned& size,
                                 const std::string& oname, 
                                 const unsigned& oindex,
                                 const bool do_tilt,
                                 const bool do_eval)
{
  GeometryAccess::shpGO geo = (oname.empty()) ? get_top_geo() : get_geo(oname, oindex);
  if(m_pbits & 32) {
    std::stringstream ss; ss << "get_pixel_coords(...) for geo:\n" << geo -> string_geo_children()
                             << "  do_tilt: " << do_tilt << "  do_eval: " << do_eval; 
    MSG(INFO, ss.str());
  }
  geo -> get_pixel_coords(X, Y, Z, size, do_tilt, do_eval);
}

//-------------------

void
GeometryAccess::get_pixel_xy_at_z(const double*& XatZ, 
                                  const double*& YatZ, 
				  unsigned& size,
                                  const double& Zplain, 
                                  const std::string& oname, 
                                  const unsigned& oindex)
{
  const double* X;
  const double* Y;
  const double* Z;
  //unsigned   size;
  const bool do_tilt=true;
  const bool do_eval=true;
  get_pixel_coords(X,Y,Z,size,oname,oindex,do_tilt,do_eval);

  if (!size) {
    MSG(ERROR, "get_pixel_coords returns ZERO size coordinate array...");
    abort(); 
  }

  if (p_XatZ) delete [] p_XatZ;
  if (p_YatZ) delete [] p_YatZ;

  p_XatZ = new double[size];
  p_YatZ = new double[size];

  // find Z0 as average Z if Zplain is not specified
  double Z0 = 0;
  if (Zplain) Z0 = Zplain;
  else {
    for(unsigned i=0; i<size; ++i) Z0 += Z[i]; 
    Z0 = Z0/size;
  }

  if(fabs(Z0) < 1000) {  
    XatZ = X;
    YatZ = Y;
    return;
  }  

  double R;
  for(unsigned i=0; i<size; ++i) {
    if (Z[i]) {
      R = Z0/Z[i];
      p_XatZ[i] = X[i]*R;
      p_YatZ[i] = Y[i]*R;
    } else {
      p_XatZ[i] = X[i];
      p_YatZ[i] = Y[i];
    }
  }
  XatZ = p_XatZ;
  YatZ = p_YatZ;
}

//-------------------

void
GeometryAccess::get_pixel_areas(const double*& A, 
				unsigned& size,
                                const std::string& oname, 
                                const unsigned& oindex)
{
  GeometryAccess::shpGO geo = (oname.empty()) ? get_top_geo() : get_geo(oname, oindex);
  if(m_pbits & 32) {
    std::string msg = "get_pixel_areas(...) for geo:\n" + geo -> string_geo_children();
    MSG(INFO, msg);
  }
  geo -> get_pixel_areas(A, size);
}

//-------------------

void
GeometryAccess::get_pixel_mask(const int*& mask, 
			       unsigned& size,
                               const std::string& oname, 
                               const unsigned& oindex,
                               const unsigned& mbits)
{
  //cout << "GeometryAccess::get_pixel_mask(): mbits =" << mbits << '\n';   

  GeometryAccess::shpGO geo = (oname.empty()) ? get_top_geo() : get_geo(oname, oindex);
  if(m_pbits & 32) {
    std::string msg = "get_pixel_areas(...) for geo:\n" + geo -> string_geo_children();
    MSG(INFO, msg);
  }
  geo -> get_pixel_mask(mask, size, mbits);
}

//-------------------

double
GeometryAccess::get_pixel_scale_size(const std::string& oname, 
                                     const unsigned& oindex)
{
  GeometryAccess::shpGO geo = (oname.empty()) ? get_top_geo() : get_geo(oname, oindex);
  return geo -> get_pixel_scale_size();
}

//-------------------

void
GeometryAccess::set_geo_pars(const std::string& oname, 
		             const unsigned& oindex,
                             const double& x0,
                             const double& y0,
                             const double& z0,
                             const double& rot_z,
                             const double& rot_y,
                             const double& rot_x,                  
                             const double& tilt_z,
                             const double& tilt_y,
                             const double& tilt_x 
		             )
{
  GeometryAccess::shpGO geo = (oname.empty()) ? get_top_geo() : get_geo(oname, oindex);
  geo -> set_geo_pars(x0, y0, z0, rot_z, rot_y, rot_x, tilt_z, tilt_y, tilt_x);
}

//-------------------

void
GeometryAccess::move_geo(const std::string& oname, 
		         const unsigned& oindex,
                         const double& dx,
                         const double& dy,
                         const double& dz
			 )
{
  GeometryAccess::shpGO geo = (oname.empty()) ? get_top_geo() : get_geo(oname, oindex);
  geo -> move_geo(dx, dy, dz);
}

//-------------------

void
GeometryAccess::tilt_geo(const std::string& oname, 
		         const unsigned& oindex,
                         const double& dt_x,
                         const double& dt_y,
                         const double& dt_z
			 )
{
  GeometryAccess::shpGO geo = (oname.empty()) ? get_top_geo() : get_geo(oname, oindex);
  geo -> tilt_geo(dt_x, dt_y, dt_z);
}

//-------------------

void GeometryAccess::print_list_of_geos()
{
  std::stringstream ss; ss << "print_list_of_geos():";
  if( v_list_of_geos.empty() ) ss << "List of geos is empty...";
  for(std::vector<GeometryAccess::shpGO>::iterator it  = v_list_of_geos.begin(); 
                                   it != v_list_of_geos.end(); ++it) {
    ss << '\n' << (*it)->string_geo();
  }
  //std::cout << ss.str();
  MSG(INFO, ss.str());
}

//-------------------

void GeometryAccess::print_list_of_geos_children()
{
  std::stringstream ss; ss << "print_list_of_geos_children(): ";
  if( v_list_of_geos.empty() ) ss << "List of geos is empty...";

  for(std::vector<GeometryAccess::shpGO>::iterator it  = v_list_of_geos.begin(); 
                                   it != v_list_of_geos.end(); ++it) {
    ss << '\n' << (*it)->string_geo_children();
  }
  //std::cout << ss.str() << '\n';
  MSG(INFO, ss.str());
}

//-------------------

void GeometryAccess::print_comments_from_dict()
{ 
  std::stringstream ss; ss << "print_comments_from_dict():\n"; 

  std::map<int, std::string>::iterator iter;

  for (iter = m_dict_of_comments.begin(); iter != m_dict_of_comments.end(); ++iter) {
    ss << "  key:" << std::setw(4) << std::left << iter->first;
    ss << "  val:" << iter->second << '\n';
  }
  //std::cout << ss.str();
  MSG(INFO, ss.str());
}
//-------------------

void
GeometryAccess::print_pixel_coords(const std::string& oname, 
                                   const unsigned& oindex)
{
  const double* X;
  const double* Y;
  const double* Z;
  unsigned   size;
  const bool do_tilt=true;
  get_pixel_coords(X,Y,Z,size,oname,oindex,do_tilt);

  std::stringstream ss; ss << "print_pixel_coords():\n"
			   << "size=" << size << '\n' << std::fixed << std::setprecision(1);  
  ss << "X: "; for(unsigned i=0; i<10; ++i) ss << std::setw(10) << X[i] << ", "; ss << "...\n";
  ss << "Y: "; for(unsigned i=0; i<10; ++i) ss << std::setw(10) << Y[i] << ", "; ss << "...\n"; 
  ss << "Z: "; for(unsigned i=0; i<10; ++i) ss << std::setw(10) << Z[i] << ", "; ss << "...\n"; 
  //cout << ss.str();
  MSG(INFO, ss.str());
}

//-------------------
//-------------------

void
GeometryAccess::get_pixel_coord_indexes(const unsigned *& iX, 
                                        const unsigned *& iY, 
				        unsigned& size,
                                        const std::string& oname, 
					const unsigned& oindex, 
                                        const double& pix_scale_size_um, 
                                        const int* xy0_off_pix,
                                        const bool do_tilt)
{
  const double* X;
  const double* Y;
  const double* Z;

  get_pixel_coords(X,Y,Z,size,oname,oindex,do_tilt);
  
  double pix_size = (pix_scale_size_um) ? pix_scale_size_um : get_pixel_scale_size(oname, oindex);

  if (p_iX) delete [] p_iX;
  if (p_iY) delete [] p_iY;

  p_iX = new unsigned[size];
  p_iY = new unsigned[size];

  if (xy0_off_pix) {
    // Offset in pix -> um
    double x_off_um = xy0_off_pix[0] * pix_size;
    double y_off_um = xy0_off_pix[1] * pix_size;
    // Protection against wrong offset bringing negative indexes
    double x_min=0; for(unsigned i=0; i<size; ++i) { if (X[i] + x_off_um < x_min) x_min = X[i] + x_off_um; } x_off_um -= x_min - pix_size/2;
    double y_min=0; for(unsigned i=0; i<size; ++i) { if (Y[i] + y_off_um < y_min) y_min = Y[i] + y_off_um; } y_off_um -= y_min - pix_size/2;

    for(unsigned i=0; i<size; ++i) { 
      p_iX[i] = (unsigned)((X[i] + x_off_um) / pix_size);
      p_iY[i] = (unsigned)((Y[i] + y_off_um) / pix_size);
    }
  } 
  else {
    // Find coordinate min values
    double x_min=X[0]; for(unsigned i=0; i<size; ++i) { if (X[i] < x_min) x_min = X[i]; } x_min -= pix_size/2;
    double y_min=Y[0]; for(unsigned i=0; i<size; ++i) { if (Y[i] < y_min) y_min = Y[i]; } y_min -= pix_size/2;
    for(unsigned i=0; i<size; ++i) { 
      p_iX[i] = (unsigned)((X[i] - x_min) / pix_size);
      p_iY[i] = (unsigned)((Y[i] - y_min) / pix_size);
    }
  }

  iX = p_iX;
  iY = p_iY;
}

//-------------------

void
GeometryAccess::get_pixel_xy_inds_at_z(const unsigned *& iX, 
                                       const unsigned *& iY, 
				       unsigned& size,
                                       const double& Zplain, 
                                       const std::string& oname, 
				       const unsigned& oindex, 
                                       const double& pix_scale_size_um, 
                                       const int* xy0_off_pix)
{
  const double* X;
  const double* Y;

  get_pixel_xy_at_z(X,Y,size,Zplain,oname,oindex);
  
  double pix_size = (pix_scale_size_um) ? pix_scale_size_um : get_pixel_scale_size(oname, oindex);

  if (p_iX) delete [] p_iX;
  if (p_iY) delete [] p_iY;

  p_iX = new unsigned[size];
  p_iY = new unsigned[size];

  if (xy0_off_pix) {
    // Offset in pix -> um
    double x_off_um = xy0_off_pix[0] * pix_size;
    double y_off_um = xy0_off_pix[1] * pix_size;
    // Protection against wrong offset bringing negative indexes
    double x_min=0; for(unsigned i=0; i<size; ++i) { if (X[i] + x_off_um < x_min) x_min = X[i] + x_off_um; } x_off_um -= x_min - pix_size/2;
    double y_min=0; for(unsigned i=0; i<size; ++i) { if (Y[i] + y_off_um < y_min) y_min = Y[i] + y_off_um; } y_off_um -= y_min - pix_size/2;

    for(unsigned i=0; i<size; ++i) { 
      p_iX[i] = (unsigned)((X[i] + x_off_um) / pix_size);
      p_iY[i] = (unsigned)((Y[i] + y_off_um) / pix_size);
    }
  } 
  else {
    // Find coordinate min values
    double x_min=X[0]; for(unsigned i=0; i<size; ++i) { if (X[i] < x_min) x_min = X[i]; } x_min -= pix_size/2;
    double y_min=Y[0]; for(unsigned i=0; i<size; ++i) { if (Y[i] < y_min) y_min = Y[i]; } y_min -= pix_size/2;
    for(unsigned i=0; i<size; ++i) { 
      p_iX[i] = (unsigned)((X[i] - x_min) / pix_size);
      p_iY[i] = (unsigned)((Y[i] - y_min) / pix_size);
    }
  }

  iX = p_iX;
  iY = p_iY;
}

//-------------------
//-------------------

NDArray<GeometryAccess::image_t> &
GeometryAccess::ref_img_from_pixel_arrays(const unsigned*& iX, 
                                          const unsigned*& iY, 
                                          const double*    W,
                                          const unsigned&  size)
{
    unsigned ix_max=iX[0]; for(unsigned i=0; i<size; ++i) { if (iX[i] > ix_max) ix_max = iX[i]; } ix_max++;
    unsigned iy_max=iY[0]; for(unsigned i=0; i<size; ++i) { if (iY[i] > iy_max) iy_max = iY[i]; } iy_max++;

    if(p_image) delete p_image;

    unsigned shape[2] = {ix_max, iy_max};
    p_image = new NDArray<GeometryAccess::image_t> (shape);
    NDArray<GeometryAccess::image_t> img = *p_image;
    // std::fill_n(img, int(img.size()), GeometryAccess::image_t(0));
    for(NDArray<GeometryAccess::image_t>::iterator it=img.begin(); it!=img.end(); it++) { *it = 0; }

    if (W) for(unsigned i=0; i<size; ++i) img(iX[i],iY[i]) = (GeometryAccess::image_t) W[i];
    else   for(unsigned i=0; i<size; ++i) img(iX[i],iY[i]) = 1;
    return *p_image;
}

//-------------------
//-------------------
//-- Static Methods--
//-------------------
//-------------------

NDArray<GeometryAccess::image_t>
GeometryAccess::img_from_pixel_arrays(const unsigned*& iX, 
                                      const unsigned*& iY, 
                                      const double*    W,
                                      const unsigned&  size)
{
    unsigned ix_max=iX[0]; for(unsigned i=0; i<size; ++i) { if (iX[i] > ix_max) ix_max = iX[i]; } ix_max++;
    unsigned iy_max=iY[0]; for(unsigned i=0; i<size; ++i) { if (iY[i] > iy_max) iy_max = iY[i]; } iy_max++;

    NDArray<GeometryAccess::image_t> img = make_NDArray<GeometryAccess::image_t>(ix_max, iy_max);
    // std::fill_n(img, int(img.size()), GeometryAccess::image_t(0));
    for(NDArray<GeometryAccess::image_t>::iterator it=img.begin(); it!=img.end(); it++) { *it = 0; }

    if (W) for(unsigned i=0; i<size; ++i) img(iX[i],iY[i]) = (GeometryAccess::image_t) W[i];
    else   for(unsigned i=0; i<size; ++i) img(iX[i],iY[i]) = 1;
    return img;
}

//-------------------

} // namespace psalg

//-------------------
