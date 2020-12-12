#include "ShmemClient.hh"

using namespace psalg::shmem;

/*
** ++
**
**
** --
*/

void ShmemClient::free(int index, size_t size)
{
}

/*
** ++
**
**
** --
*/

void* ShmemClient::get(int& index, size_t& size)
{
  return 0;
}

int ShmemClient::connect(const char* tag, int tr_index)
{
  return 1;
}
