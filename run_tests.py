#!/usr/bin/env python
"""
Test runner script for AI Investment Agent

This script runs all tests with coverage reporting and generates a summary.

Usage:
    python run_tests.py [--html] [--xml] [--verbose]

Options:
    --html      Generate HTML coverage report
    --xml       Generate XML coverage report
    --verbose   Run tests in verbose mode
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run tests for AI Investment Agent")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--xml", action="store_true", help="Generate XML coverage report")
    parser.add_argument("--verbose", action="store_true", help="Run tests in verbose mode")
    return parser.parse_args()

def run_tests(args):
    """Run tests with coverage"""
    print("\n" + "=" * 80)
    print(f"Running tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Build the pytest command
    cmd = ["pytest"]
    
    # Add coverage options
    cmd.extend(["--cov=src/", "--cov-report=term"])
    
    if args.html:
        cmd.append("--cov-report=html")
    
    if args.xml:
        cmd.append("--cov-report=xml")
    
    if args.verbose:
        cmd.append("-v")
    
    # Add test directories
    cmd.append("tests/")
    
    # Run the command
    result = subprocess.run(cmd)
    
    return result.returncode

def print_summary(exit_code):
    """Print test summary"""
    print("\n" + "=" * 80)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code {exit_code}")
    print("=" * 80)
    
    # Check if coverage reports were generated
    if os.path.exists("htmlcov"):
        print("HTML coverage report generated in 'htmlcov' directory")
    
    if os.path.exists("coverage.xml"):
        print("XML coverage report generated as 'coverage.xml'")

def main():
    """Main function"""
    args = parse_args()
    exit_code = run_tests(args)
    print_summary(exit_code)
    return exit_code

if __name__ == "__main__":
    sys.exit(main())