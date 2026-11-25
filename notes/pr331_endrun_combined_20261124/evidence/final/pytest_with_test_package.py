import sys,types
import psana
package=types.ModuleType('psana.tests')
package.__path__=['/pscratch/sd/m/monarin/agent-work/lcls2/pr331-endrun-combined-20261124-attempt01/source/psana/psana/tests']
package.__package__='psana.tests'
sys.modules['psana.tests']=package
psana.tests=package
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
