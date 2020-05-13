import pytest

@pytest.mark.skip(reason="waiting for move to ps-3.1.7")
def test_rogue():
    # do this simple test since rogue has a rather complex boost dependency
    # which we have gotten wrong in the past - valmar and cpo
    import rogue

if __name__ == "__main__":
    test_rogue()
