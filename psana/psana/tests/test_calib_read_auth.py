"""Public calibration reads must work independently of authentication state."""

import importlib.util
from types import SimpleNamespace

import pytest

from psana.pscalib.calib import CalibConstants as cc
from psana.pscalib.calib import MDBWebUtils as wu


@pytest.mark.parametrize('has_jwt', [False, True])
@pytest.mark.parametrize('path', ['/ws/calib_ws', '/ws/calib_ws/db/coll', '/calib_ws/db/coll'])
def test_public_read_does_not_authenticate(monkeypatch, has_jwt, path):
    def unexpected_auth(*args, **kwargs):
        pytest.fail('Public reads must not use Kerberos or the JWT session')

    calls = []
    response = object()

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(wu, 'has_jwt', has_jwt)
    monkeypatch.setattr(wu, 'USE_QUERY_STR', not has_jwt)
    monkeypatch.setattr(wu, 'session', SimpleNamespace(get=unexpected_auth))
    monkeypatch.setattr(cc, 'krbheaders', unexpected_auth)
    monkeypatch.setattr(wu.req, 'get', get)
    url = 'https://pswww.slac.stanford.edu' + path
    query = {'query_string': '{"ctype":"pedestals"}'} if not has_jwt else {'ctype': 'pedestals'}

    assert wu.get(url, query=query, timeout=7) is response
    query_arg = 'json' if has_jwt else 'params'
    assert calls == [(url, {query_arg: query, 'timeout': 7})]


@pytest.mark.parametrize('has_jwt', [False, True])
@pytest.mark.parametrize('method', ['get', 'post', 'put', 'delete'])
def test_authenticated_operations_keep_credentials(monkeypatch, has_jwt, method):
    calls = []
    response = SimpleNamespace(ok=True, text='ok')
    headers = {'Authorization': 'test-kerberos-credential'}

    def kerberos_headers():
        calls.append('kerberos')
        return headers

    def jwt_request(url, **kwargs):
        calls.append('jwt')
        assert 'headers' not in kwargs  # Bearer header belongs to the session.
        return response

    def kerberos_request(url, **kwargs):
        assert kwargs['headers'] == headers
        calls.append('request')
        return response

    monkeypatch.setattr(wu, 'has_jwt', has_jwt)
    monkeypatch.setattr(wu, 'session', SimpleNamespace(**{method: jwt_request}))
    monkeypatch.setattr(cc, 'krbheaders', kerberos_headers)
    monkeypatch.setattr(wu.req, method, kerberos_request)
    url = 'https://pswww.slac.stanford.edu/ws-%s/calib_ws/db/coll' % ('jwt' if has_jwt else 'kerb')

    if method == 'delete':
        actual = wu.delete_cmd(url)
    elif method in ('post', 'put'):
        actual = getattr(wu, method)(url, doc={'run': 14})
    else:
        actual = wu.get(url)
    assert actual is response
    assert calls == (['jwt'] if has_jwt else ['kerberos', 'request'])


def test_legacy_headers_are_created_only_on_access(monkeypatch):
    import krtc

    calls = []

    def ticket(service):
        calls.append(service)
        return SimpleNamespace(getAuthHeaders=lambda: {'Authorization': 'test-credential'})

    monkeypatch.setattr(krtc, 'KerberosTicket', ticket)
    spec = importlib.util.spec_from_file_location('calib_constants_auth_test', cc.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert calls == []
    assert module.KRBHEADERS['Authorization'] == 'test-credential'
    assert calls == ['HTTP@pswww.slac.stanford.edu']
    with pytest.raises(AttributeError):
        module.nonexistent_attribute
