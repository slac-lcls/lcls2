#ifndef AxiMicronN25Q_hh
#define AxiMicronN25Q_hh

class AxiMicronN25Q {
public:
  AxiMicronN25Q(char* base, const char* fname);
  ~AxiMicronN25Q();
public:
  void load();
  void verify();
private:
  class PrivateData;
  PrivateData* _private;
};

#endif
