#ifndef CTEST_NDA_H
#define CTEST_NDA_H

//#template <typename T>
//ctest_nda<T>(T *arr, int r, int c) {
#include <unistd.h> // size_t, ssize_t, int16_t, ...

#include <iostream>
using namespace std;

//-----------------------------

template <typename T>
void ctest_nda(T *arr, int r, int c) {
  cout << "In ctest_nda r=" << r << " c=" << c << " arr: ";
  for(int i=0; i < r*c; i++) {
      if (i<10) cout << arr[i] << ' '; 
      arr[i] += i;
  } cout << '\n';
}

//-----------------------------
void ctest_nda_f8(double   *arr, int r, int c) { ctest_nda<double>  (arr, r, c); }
void ctest_nda_i2(int16_t  *arr, int r, int c) { ctest_nda<int16_t> (arr, r, c); }
void ctest_nda_u2(uint16_t *arr, int r, int c) { ctest_nda<uint16_t>(arr, r, c); }
//-----------------------------

#endif

