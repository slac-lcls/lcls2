/**
 * @file
 * @brief Declares small test helpers (in namespace psalg) that print and modify arrays and vectors.
 */
#ifndef CTEST_UTILS_H
#define CTEST_UTILS_H

//#template <typename T>
//ctest_nda<T>(T *arr, int r, int c) {

#include <unistd.h> // size_t, ssize_t, int16_t, ...
#include <vector>
#include <iostream>

using namespace std;
//-----------------------------

namespace psalg {

/** Test helper: print ndim and the first ndim entries of sh, then print the first 5 elements of arr and add each element's index to it (always 5 elements). */
template <typename T>
void ctest_nda_v2(T *arr, uint32_t* sh, int& ndim) {
  cout << "In ctest_nda ndim=" << ndim << " shape: ";
  for(int i=0; i<ndim; i++) {cout << sh[i] << ' ';} 
  cout << "\n arr: ";
  for(int i=0; i<5; i++) {
      cout << arr[i] << ' ';
      arr[i] = i + arr[i];
  } cout << '\n';
}

//-----------------------------

/** Test helper: print r, c and the r*c elements of arr to stdout, adding each element's index to it (arr[i] = i + arr[i]). */
template <typename T>
void ctest_nda(T *arr, int r, int c) {
  cout << "In ctest_nda r=" << r << " c=" << c << " arr: ";
  for(int i=0; i < r*c; i++) {
      cout << arr[i] << ' '; 
      arr[i] = i + arr[i];
  } cout << '\n';
}

//-----------------------------
/** Call ctest_nda<double>(arr, r, c). */
void ctest_nda_f8(double   *arr, int r, int c) { ctest_nda<double>  (arr, r, c); }
/** Call ctest_nda<int16_t>(arr, r, c). */
void ctest_nda_i2(int16_t  *arr, int r, int c) { ctest_nda<int16_t> (arr, r, c); }
/** Call ctest_nda<uint16_t>(arr, r, c). */
void ctest_nda_u2(uint16_t *arr, int r, int c) { ctest_nda<uint16_t>(arr, r, c); }
//-----------------------------

/** Test helper: print the size of v and its first 5 elements (not range-checked) to stdout. */
template <typename T>
void ctest_vector(vector<T>& v) {
  cout << "In ctest_vector size: " << v.size() << " v=" ;
  for(int i=0; i<5; i++) {cout << v[i] << ' ';}
  cout << '\n';
}

}; //namespace psalg

//-----------------------------
//-----------------------------
//-----------------------------

#endif

