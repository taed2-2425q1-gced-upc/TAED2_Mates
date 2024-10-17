Overview

This directory contains unit tests, integration tests, and any other automated tests for the project.

- Include tests for each module in the `mates/` folder.
- Use testing frameworks to ensure all functionalities are working as expected.


Usage

- **Running All Tests**: 
    ```bash
    pytest
    ```

- **Running Specific Tests**: 
    ```bash
    pytest tests/test_file_name.py
    ```

- **Test Coverage** (will provide a report on which parts of the code are covered by the tests):
    ```bash
    pytest --cov=mates tests/
    ```

- **Debugging Failed Tests**: if any tests fail, pytest will show detailed error messages. Review those messages to debug the issue and fix any bugs in the code or tests.

- **Writing New Tests**:
  - Add new test files following the naming convention `test_*.py`.
  - Keep test files organized by module or feature being tested.
