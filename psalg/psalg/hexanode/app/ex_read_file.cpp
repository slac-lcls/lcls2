#include <stdio.h>
#include <iostream>  // cout
#include <fstream>  // ifstream




int main_v0 ()
{
   FILE *fp;
   char c;

   int counter=0;
  
   fp = fopen("hexanode-example-CO_4.lmf","r");
   while(counter<320000)
     //while(true)
   {
      counter++;
      c = fgetc(fp);
      if( feof(fp) )
      { 
         break;
      }
      printf("%c", c);

      //std::cout << v << '\n';
   }
   fclose(fp);
   
   return(0);
}

int main()
{
  std::string fname = "hexanode-example-CO_4.lmf";

   std::ifstream in(fname.c_str());
   if (not in.good()) { 
     std::cout << "Failed to open file: " << fname << '\n'; 
     return(1);
   }
 
   int counter=0;
   std::string str; 
   while(getline(in,str)) {
     counter++;
     if(counter>9250) break;
     std::cout << str << '\n';    
   }

   in.close();
 
   return(0);
}


