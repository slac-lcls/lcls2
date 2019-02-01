#! /bin/bash

echo "Clean lcls2 repository from files genetated during build" 

REPODIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo REPODIR: $REPODIR

echo rm -rf $REPODIR/install
     rm -rf $REPODIR/install

echo cd $REPODIR/psana
     cd $REPODIR/psana
echo rm -rf {build,*.egg-info,*.so,psana/*.so}
     rm -rf {build,*.egg-info,*.so,psana/*.so}

echo rm -rf $REPODIR/psalg/build
     rm -rf $REPODIR/psalg/build

echo rm -rf $REPODIR/psdaq/build
     rm -rf $REPODIR/psdaq/build

echo rm -rf $REPODIR/xtcdata/build
     rm -rf $REPODIR/xtcdata/build

#echo cd $REPODIR/psana/psana
#     cd $REPODIR/psana/psana
#echo rm {bufferedreader.c,hsd/hsd.cpp,peakFinder/dgramCreate.cpp,peakFinder/peakFinder.cpp,smdreader.c,../src/dgramchunk.c}
#     rm {bufferedreader.c,hsd/hsd.cpp,peakFinder/dgramCreate.cpp,peakFinder/peakFinder.cpp,smdreader.c,../src/dgramchunk.c}

echo find . -name "*~" -delete
     find . -name "*~" -delete

echo find . -name "*.pyc" -delete
     find . -name "*.pyc" -delete

echo find . -name __pycache__ -type d -exec rm -rf {} +
     find . -name __pycache__ -type d -exec rm -rf {} +

echo "Cleaning is done"
