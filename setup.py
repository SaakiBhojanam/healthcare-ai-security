#!/usr/bin/env python3
"""
Security for Health AI Platform Setup Script
Automated installation and startup for development and deployment
"""

import subprocess
import sys
import os
import time

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed: {e.stderr}")
        return False

def check_python_version():
    """Verify Python 3.8+ is installed"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}")
        return False

def main():
    print("Security for Health AI Platform - Setup Script")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install dependencies
    setup_steps = [
        ("pip install -r requirements.txt", "Installing Python dependencies"),
        ("pip install -r backend/requirements.txt", "Installing backend-specific dependencies"),
    ]

    for command, description in setup_steps:
        if not run_command(command, description):
            print("Setup failed. Please check the error messages above.")
            sys.exit(1)

    print("\nSetup completed successfully!")
    print("\nNext Steps:")
    print("1. Start the API server:")
    print("   cd backend && python3 simple_main.py")
    print("\n2. Run tests:")
    print("   python3 test_api.py")
    print("\n3. View API documentation:")
    print("   http://localhost:8000/docs")

    # Ask if user wants to start the server
    response = input("\n Start the API server now? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        os.chdir('backend')
        print("\nStarting Security for Health AI Platform API...")
        print("API Documentation: http://localhost:8000/docs")
        print("Security Dashboard: http://localhost:8000/ai/security-dashboard")
        print("\nPress Ctrl+C to stop the server\n")
        subprocess.run(["python3", "simple_main.py"])

if __name__ == "__main__":
    main()
