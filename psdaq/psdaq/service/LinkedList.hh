/**
 * @file
 * @brief ListBase and LinkedList: intrusive doubly linked lists where an object serves either as a list head or as an element.
 */
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
/** Node of an intrusive doubly linked list; an object can be a list head (label) or an element. No locking (see LinkedListSL for a locked version). */
class ListBase
  {
  public:
    /** Does nothing; empty body. */
    ~ListBase();
    /** Construct as an empty list head: both links point to itself. */
    ListBase();
    /** Append this object at the tail of the list whose head is listhead. */
    ListBase(ListBase* listhead);
    /** Insert this object after after and return after. */
    ListBase* connect(ListBase* after);
    /** Unlink this object from its list and return it (its own links are left unchanged). */
    ListBase* disconnect();
    /** Treating this object as a list head, append entry at the tail; returns the previous tail. */
    ListBase* insert(ListBase*);
    /** Treating this object and entry as list heads, move all elements of entry to the tail of this list, leaving entry empty; returns this. */
    ListBase* insertList(ListBase*);
    /** Treating this object as a list head, unlink and return the head element (the head itself, i.e. empty(), if the list is empty). */
    ListBase* remove();
    /** Return the address of this object as the list label, which forward() returns when the list is empty. */
    ListBase* empty()  const;
    /** Return the next node (the head element when this is a list head). */
    ListBase* forward() const;
    /** Return the previous node (the tail element when this is a list head). */
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
  Pds::ListBase* next = _flink;
  Pds::ListBase* prev = _blink;
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
/** Typed wrapper over ListBase whose operations return T pointers. */
template<class T>
class LinkedList : public ListBase
  {
  public:
    /** Does nothing; empty body. */
    ~LinkedList(){}
    /** Construct as an empty list head. */
    LinkedList() :            ListBase()         {}
    /** Append this object at the tail of the list whose head is listhead. */
    LinkedList(T* listhead) : ListBase(listhead) {}
    /** Insert this object after after; returns after as T*. */
    T* connect(T* after)          {return (T*)ListBase::connect(after);}
    /** Unlink this object and return it as T*. */
    T* disconnect()               {return (T*)ListBase::disconnect();}
    /** Append entry at the tail of this list head; returns the previous tail as T*. */
    T* insert(ListBase* entry)    {return (T*)ListBase::insert(entry);}
    /** Move all elements of list to the tail of this list; returns this as T*. */
    T* insertList(ListBase* list) {return (T*)ListBase::insertList(list);}
    /** Unlink and return the head element as T* (empty() if none). */
    T* remove()                   {return (T*)ListBase::remove();}
    /** Return the list label as T*. */
    T* empty()  const             {return (T*)ListBase::empty();}
    /** Return the next node as T*. */
    T* forward() const            {return (T*)ListBase::forward();}
    /** Return the previous node as T*. */
    T* reverse() const            {return (T*)ListBase::reverse();}
  };
}
#endif
