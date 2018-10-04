echo "=== Running tests ==="
cd ..
pytest --cov=pyDSA --cov-report=html
$BROWSER htmlcov/index.html &
