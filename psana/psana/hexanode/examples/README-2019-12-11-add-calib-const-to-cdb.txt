
2019-12-11 Add quadanode calibration constants to calibration DB
================================================================

need in calib constants for test data:
/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2

cdb - prints info about command

cdb add -e amox27716 -d tmo_quadanode -c calibcfg -r 100 -f configuration_quad.txt     -i txt -u dubrovin
cdb add -e amox27716 -d tmo_quadanode -c calibtab -r 100 -f calibration_table_data.txt -i txt -u dubrovin

-------------------------------

https://github.com/slac-lcls/lcls2/blob/master/psalg/psalg/hexanode/SortUtils.hh
https://github.com/slac-lcls/lcls2/blob/master/psalg/psalg/hexanode/src/SortUtils.cc



Two methods need to be extended for case of parsing text in stead of reading file

bool read_config_file(const char * name, sort_class *& sorter, int& command,
                      double& offset_sum_u, double& offset_sum_v, double& offset_sum_w,
                      double& w_offset, double& pos_offset_x, double& pos_offset_y);

bool read_calibration_tables(const char * filename, sort_class * sorter);
