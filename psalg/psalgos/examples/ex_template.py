#------------------------------

def ex_01(ntest) : 
    from ex_source_dsname import ex_source_dsname

    src, dsn = ex_source_dsname(1)
    print 'src=%s, dsname=%s' % (src, dsn)

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print 50*'_', '\nTest %s:' % tname
    if   tname == '0' : ex_01(tname);
    elif tname == '1' : ex_01(tname)
    else : print 'Not-recognized test name: %s' % tname
    sys.exit('End of test %s' % tname)
 
#------------------------------
