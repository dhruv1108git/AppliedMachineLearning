pytest -v test.py
pytest -v --cov=score --cov=test --cov-report=term --cov-report=term-missing test.py > coverage.txt