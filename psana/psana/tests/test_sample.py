""" run it as
    pytest psana/psana/tests/ # in your git root directory or
    pytest test_sample.py
"""

def func(x):
    return x + 1

def test_answer():
    assert func(3) == 4

# EOF
