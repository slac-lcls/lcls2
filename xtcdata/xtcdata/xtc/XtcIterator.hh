/**
 * @file
 * @brief Declares XtcData::XtcIterator, the abstract base class for walking the child xtcs of an Xtc.
 */
#ifndef PDS_XTCITERATOR
#define PDS_XTCITERATOR

/*
** ++
**  Package:
**	OdfContainer
**
**  Abstract:
**      This class allows iteration over a collection of "odfInXtcs".
**      An "event" generated from DataFlow consists of data described
**      by a collection of "odfInXtcs". Therefore, this class is
**      instrumental in the navigation of an event's data. The set of
**      "odfInXtcs" is determined by passing (typically to the constructor)
**      a root "odfInXtc" which describes the collection of "odfInXtcs"
**      to process. This root, for example is provided by an event's
**      datagram. As this is an Abstract-Base-Class, it is expected that an
**      application will subclass from this class, providing an implementation
**      of the "process" method. This method will be called back for each
**      "odfInXtc" in the collection. Note that the "odfInXtc" to process
**      is passed as an argument. If the "process" method wishes to abort
**      the iteration a zero (0) value is returned. The iteration is initiated
**      by calling the "iterate" member function.
**
**  Author:
**      Michael Huffer, SLAC, (415) 926-4269
**
**  Creation Date:
**	000 - October 11,1998
**
**  Revision History:
**	None.
**
** --
*/

namespace XtcData
{

class Xtc;

/**
 * Abstract iterator over the child xtcs of a root Xtc. iterate() calls process() once per direct child and stops when it returns 0.
 * It does not descend into children by itself; subclasses call iterate(child, bufEnd) from process() to do that.
 */
class XtcIterator
{
public:
    /** Store root and bufEnd for use by iterate(). */
    XtcIterator(Xtc* root, const void* bufEnd);
    /** Default constructor; root and bufEnd are left uninitialized (use iterate(Xtc*, const void*)). */
    XtcIterator()
    {
    }
    /** Virtual destructor; does nothing. */
    virtual ~XtcIterator()
    {
    }

public:
    /** Called by iterate() for each child xtc; return 0 to stop iterating, nonzero to continue. Pure virtual. */
    virtual int process(Xtc* xtc, const void* bufEnd) = 0;

public:
    /** Call iterate(root, bufEnd) with the values given to the constructor. */
    void iterate();
    /**
     * Call process() on each child xtc in root's payload, in order, until process() returns 0 or the payload is used up.
     * Returns at once if root's damage has the Damage::Corrupted bit set. Prints a message and calls abort() if a child starts at or past a non-null bufEnd or has extent < sizeof(Xtc).
     */
    void iterate(Xtc*, const void* bufEnd);
    /** Return the root xtc given to the constructor. */
    const Xtc* root() const;

private:
    Xtc* _root; // Collection to process in the absence of an argument...
    const void* _bufEnd;
};
}

/*
** ++
**
**    This constructor takes an argument the "Xtc" which defines the
**    collection to iterate over.
**
**
** --
*/

inline XtcData::XtcIterator::XtcIterator(Xtc* root, const void* bufEnd) : _root(root), _bufEnd(bufEnd)
{
}

/*
** ++
**
**    This function will return the collection specified by the constructor.
**
** --
*/

inline const XtcData::Xtc* XtcData::XtcIterator::root() const
{
    return _root;
}

/*
** ++
**
**    This function will commence iteration over the collection specified
**    by the constructor.
**
** --
*/

inline void XtcData::XtcIterator::iterate()
{
    iterate(_root, _bufEnd);
}

#endif
