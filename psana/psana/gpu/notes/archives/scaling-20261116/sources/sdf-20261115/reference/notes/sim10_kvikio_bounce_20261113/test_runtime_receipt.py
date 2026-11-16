import copy
import pytest
from prepare_runtime import validate_configs,ALLOWED_CONFIG_MISMATCHES

def test_accept_exact_packaging_difference():
    configs=copy.deepcopy(ALLOWED_CONFIG_MISMATCHES)
    configs['meson.build']=dict(pr='same',frozen='same')
    assert validate_configs(configs)==ALLOWED_CONFIG_MISMATCHES

@pytest.mark.parametrize('change',['missing','new','changed','absent_frozen'])
def test_reject_unapproved_config(change):
    configs=copy.deepcopy(ALLOWED_CONFIG_MISMATCHES)
    if change=='missing':configs={}
    elif change=='new':configs['meson.build']=dict(pr='new',frozen='old')
    elif change=='changed':configs['psana/meson.build']['pr']='changed'
    else:configs['meson.build']=dict(pr='new',frozen=None)
    with pytest.raises(ValueError):validate_configs(configs)
