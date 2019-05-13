#! /bin/bash

echo "Clean lcls2 repository from files genetated during build" 

REPODIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo REPODIR: $REPODIR

echo rm -rf $REPODIR/install
     rm -rf $REPODIR/install

echo rm -rf $REPODIR/psana/{build,*.egg-info,*.so,psana/*.so}
     rm -rf $REPODIR/psana/{build,*.egg-info,*.so,psana/*.so}

echo find . -name "*~" -delete
     find . -name "*~" -delete

echo find . -name "*.pyc" -delete
     find . -name "*.pyc" -delete

echo find . -name build -type d -exec rm -rf {} +
     find . -name build -type d -exec rm -rf {} +

echo find . -name __pycache__ -type d -exec rm -rf {} +
     find . -name __pycache__ -type d -exec rm -rf {} +

echo "Cleaning is done"
