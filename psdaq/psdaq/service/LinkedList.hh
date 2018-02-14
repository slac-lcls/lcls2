/*
** ++
**  Package:
**	Service
**
**  Abstract:
**	Various classes to maniplate doubly-lined lists
**
**  Author:
**      Michael Huffer, SLAC, (415) 926-4269
**
**  Creation Date:
**	000 - June 20 1,1997
**
**  Revision History:
**	None.
**
** --
*/

#ifndef PDS_LINKEDLIST
#define PDS_LINKEDLIST

namespace Pds {
class ListBase
  {
  public:
    ~ListBase();
    ListBase(); 
    ListBase(ListBase* listhead); 
    ListBase* connect(ListBase* after);
    ListBase* disconnect();
    ListBase* insert(ListBase*);
    ListBase* insertList(ListBase*);
    ListBase* remove();
    ListBase* empty()  const;   
    ListBase* forward() const;
    ListBase* reverse() const;
  private:
    ListBase* _flink;
    ListBase* _blink;
  };
}
/*
** ++
**
**
** --
*/

inline Pds::ListBase::~ListBase(){}

/*
** ++
**
** Return the queue's label (usefull for checking for an empty queue)
**
** --
*/

inline Pds::ListBase* Pds::ListBase::empty() const 
  {
  return (ListBase*)&_flink;
  }

/*
** ++
**
**   Constructor with no arguments sets up the object as if its a listhead.
**   I.e. each link points to itself... 
**
** --
*/

inline Pds::ListBase::ListBase() :   
  _flink(empty()), 
  _blink(empty()) 
  {
  }


/*
** ++
**
**    Return the entry at the head of the doubly linked list...
**
** --
*/

inline Pds::ListBase* Pds::ListBase::forward() const 
  {
  return _flink;
  }

/*
** ++
**
**    Return the entry at the tail of the doubly linked list...
**
** --
*/

inline Pds::ListBase* Pds::ListBase::reverse() const 
  {
  return _blink;
  }

/*
** ++
**
**   Insert ourself on a doubly-linked list. The input argument is a 
**   pointer to the entry AFTER which the entry is to be inserted.
**
** --
*/

inline Pds::ListBase* Pds::ListBase::connect(ListBase* after)
  {                                                               
  Pds::ListBase* next = after->_flink;  

  _flink         = next;                        
  _blink         = after;      
  next->_blink   = this;     
  after->_flink  = this;     
  return after;
  }

/*
** ++
**
**   Remove ourself from the list we are linked to. A pointer to 
**   ourself is returned.
**
** --
*/

inline Pds::ListBase* Pds::ListBase::disconnect()    
  {                                                               
  register Pds::ListBase* next = _flink;  
  register Pds::ListBase* prev = _blink; 
  prev->_flink = next;
  next->_blink = prev;
  return this;
  }

/*
** ++
**
**   This function assumes the object represents the listhead of a 
**   doubly linked list and inserts the entry specified by the input
**   argument at the TAIL of the list.
**
** --
*/

inline Pds::ListBase* Pds::ListBase::insert(ListBase* entry)
  {                                                               
  return entry->connect(reverse());  
  }

/*
** ++
**
**   This function assumes the object and the input argument 'entry' 
**   represent the listhead of doubly linked lists.  The function
**   inserts the 'entry' list at the tail of this object's list
**   leaving the 'entry' list empty upon return.
**
** --
*/

inline Pds::ListBase* Pds::ListBase::insertList(ListBase* entry)
{                                                               
  if (entry->_flink != entry) {
    this ->_blink->_flink = entry->_flink;
    entry->_flink->_blink = this ->_blink;
    this ->_blink = entry->_blink;
    entry->_blink->_flink = this;
    entry->_flink = entry;
    entry->_blink = entry;
  }
  return this;
}

/*
** ++
**
**   Constructor with a single argument, inserts the object at the tail
**   of the list identified by the argument.
**
** --
*/

inline Pds::ListBase::ListBase(ListBase* listhead)
  {
  listhead->insert(this);  
  }

/*
** ++
**
**   This function assumes the object represents the listhead of a 
**   doubly linked list and removes the entry at the HEAD of the list.
**   The removed entry is returned to the caller.
**
** --
*/

inline Pds::ListBase* Pds::ListBase::remove()
  {                                                               
  return forward()->disconnect();  
  }

/*
** ++
**
**
** --
*/

namespace Pds {
template<class T>
class LinkedList : public ListBase
  {
  public:
    ~LinkedList(){}
    LinkedList() :            ListBase()         {} 
    LinkedList(T* listhead) : ListBase(listhead) {}
    T* connect(T* after)          {return (T*)ListBase::connect(after);}
    T* disconnect()               {return (T*)ListBase::disconnect();}
    T* insert(ListBase* entry)    {return (T*)ListBase::insert(entry);}
    T* insertList(ListBase* list) {return (T*)ListBase::insertList(list);}
    T* remove()                   {return (T*)ListBase::remove();}
    T* empty()  const             {return (T*)ListBase::empty();}
    T* forward() const            {return (T*)ListBase::forward();}
    T* reverse() const            {return (T*)ListBase::reverse();}
  };
}
#endif
