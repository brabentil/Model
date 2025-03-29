# API Testing Scripts

This directory contains scripts for testing the Fraud Detection API endpoints.

## Available Tests

- `test_transform_api.py`: Tests the `/transform` endpoint which converts raw transaction data into the feature vector format required by the model.

## Running the Tests

To run the tests, ensure you have the required dependencies installed:

```bash
pip install requests
```

Then simply run the test script:

```bash
python test_transform_api.py
```

## Adding New Tests

When adding new test scripts:

1. Follow the same pattern as existing tests
2. Ensure proper error handling
3. Include logging for debugging
4. Make tests return clear success/failure indicators
5. Add the test to this README file

## Customizing the Tests

You can customize the base URL by modifying the `BASE_URL` constant in each test script.
