#!/usr/bin/env python3
"""
Test runner script for the Wikipedia Assistant project
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Failed!")
        print(f"Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    """Main test runner"""
    print("🚀 Starting Wikipedia Assistant Test Suite")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Please run from project root.")
        sys.exit(1)
    
    # Check if Poetry is available
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: Poetry not found. Please install Poetry first.")
        sys.exit(1)
    
    # Run tests
    success = True
    
    # 1. Install dependencies
    success &= run_command("poetry install --with dev", "Installing dependencies")
    
    # 2. Run linting
    success &= run_command("poetry run python -m flake8 app scripts --count --exit-zero --max-complexity=10 --max-line-length=127", "Running linting")
    
    # 3. Run unit tests
    success &= run_command("poetry run pytest tests/ -v --cov=app --cov=scripts --cov-report=term-missing", "Running unit tests")
    
    # 4. Run integration tests
    success &= run_command("poetry run pytest tests/test_integration.py -v -m integration", "Running integration tests")
    
    # 5. Generate coverage report
    success &= run_command("poetry run pytest tests/ --cov=app --cov=scripts --cov-report=html", "Generating coverage report")
    
    # Summary
    print(f"\n{'='*50}")
    if success:
        print("🎉 All tests passed!")
        print("📊 Coverage report generated in htmlcov/")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
    print('='*50)


if __name__ == "__main__":
    main() 