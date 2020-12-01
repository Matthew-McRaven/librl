import pytest

@pytest.fixture()
def hypers():
    hypers = {}
    hypers['device'] = 'cpu'
    hypers['epochs'] = 1
    hypers['task_count'] = 1
    return hypers