/**
 * @file
 * @brief Declares small test helpers that print and modify arrays and vectors.
 */
#ifndef CTEST_UTILS_H
#define CTEST_UTILS_H

//#template <typename T>
//ctest_nda<T>(T *arr, int r, int c) {

#include <unistd.h> // size_t, ssize_t, int16_t, ...
#include <vector>
#include <iostream>

using namespace std;
//---------

namespace psalg {


//typedef int64_t NPSHAPE_T;
/** long int; shape element type used by ctest_nda_v2(). */
typedef long int NPSHAPE_T;

//---------

/** Test helper: print r, c and the r*c elements of arr to stdout, adding each element's index to it (arr[i] = i + arr[i]). */
template <typename T>
void ctest_nda(T *arr, int r, int c) {
  cout << "In ctest_nda r=" << r << " c=" << c << " arr: ";
  for(int i=0; i < r*c; i++) {
      cout << arr[i] << ' '; 
      arr[i] = i + arr[i];
  } cout << '\n';
}

//---------

/** Call ctest_nda<double>(arr, r, c). */
void ctest_nda_f8(double   *arr, int r, int c) { ctest_nda<double>  (arr, r, c); }
/** Call ctest_nda<int16_t>(arr, r, c). */
void ctest_nda_i2(int16_t  *arr, int r, int c) { ctest_nda<int16_t> (arr, r, c); }
/** Call ctest_nda<uint16_t>(arr, r, c). */
void ctest_nda_u2(uint16_t *arr, int r, int c) { ctest_nda<uint16_t>(arr, r, c); }

//---------

/** Test helper: print ndim and the first ndim entries of sh, then print the first 10 elements of arr and add 1 to each of them (always 10, whatever the shape). */
template <typename T>
void ctest_nda_v2(T *arr, NPSHAPE_T* sh, int& ndim) {
  cout << "In ctest_nda_v2 ndim=" << ndim << " shape: ";
  for(int i=0; i<ndim; i++) {cout << sh[i] << ' ';} 

  cout << "\n arr: ";
  for(int i=0; i<10; i++) {
      cout << arr[i] << ' ';
      arr[i] += 1;
  } cout << '\n';

  //sh[0]=5;
  //sh[1]=2;
}

//---------

/** Test helper: print the size of v, set v[2] to 222 (not range-checked), then print all elements of v to stdout. */
template <typename T>
void ctest_vector(vector<T>& v) {
  cout << "In ctest_vector size: " << v.size() << " v=" ;
  //for(int i=0; i<5; i++) {cout << v[i] << ' ';}
  v[2]=222; // DOES NOT RETURN IT BACK !!!!
  for(typename vector<T>::const_iterator i=v.begin(); i<v.end(); ++i) std::cout << *i << ' ';
  cout << '\n';
}

}; //namespace psalg

//---------
//---------
//---------

#endif

