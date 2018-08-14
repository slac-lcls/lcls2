/* 
 * https://curl.haxx.se/libcurl/c/getinfo.html
 */

//#include "psalg/utils/Logger.hh" // MSG, LOGGER,...

#include <stdio.h>
#include <curl/curl.h>

//-------------------
 
int test_getinfo(void)
{
  CURL *curl;
  CURLcode res;
 
  curl = curl_easy_init();
  if(curl) {
    curl_easy_setopt(curl, CURLOPT_URL, "http://www.example.com/");
    res = curl_easy_perform(curl);
 
    if(CURLE_OK == res) {
      char *ct;
      /* ask for the content-type */ 
      res = curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &ct);
 
      if((CURLE_OK == res) && ct)
        printf("We received Content-Type: %s\n", ct);
    }
 
    /* always cleanup */ 
    curl_easy_cleanup(curl);
  }
  return 0;
}

//-------------------

int main(void)
{
  //MSG(INFO, LOGGER.tstampStart() << " Logger started"); // optional record
  //LOGGER.setLogger(LL::DEBUG, "%H:%M:%S.%f");           // set level and time format

  test_getinfo();
}

//-------------------

