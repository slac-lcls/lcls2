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

class XtcIterator
{
public:
    XtcIterator(Xtc* root, const void* bufEnd);
    XtcIterator()
    {
    }
    virtual ~XtcIterator()
    {
    }

public:
    virtual int process(Xtc* xtc, const void* bufEnd) = 0;

public:
    void iterate();
    void iterate(Xtc*, const void* bufEnd);
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
