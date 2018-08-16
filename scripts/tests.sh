echo "=== Running tests ==="
cd ..
pytest --cov=PyDSA --cov-report=html
$BROWSER htmlcov/index.html &
