echo "=== Running tests ==="
cd ..
pytest --cov=pyDSA_core --cov-report=html
$BROWSER htmlcov/index.html &
