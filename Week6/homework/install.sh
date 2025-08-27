#!/bin/bash

# Voice Agent Installation Script
# This script installs dependencies for homework6.py on Linux server

echo "=========================================="
echo "Installing Voice Agent Dependencies"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "homework6.py" ]; then
    echo "Error: homework6.py not found in current directory"
    echo "Please run this script from the homework directory"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found"
    exit 1
fi

# Check if constraints.txt exists
if [ ! -f "constraints.txt" ]; then
    echo "Error: constraints.txt not found"
    exit 1
fi

echo "Installing packages with constraints..."
echo "Note: PyTorch will NOT be installed (using server's existing installation)"
echo ""

# Install with constraints
pip install -r requirements.txt -c constraints.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Installation completed successfully!"
    echo "=========================================="
    echo "You can now run the voice agent with:"
    echo "python homework6.py"
else
    echo ""
    echo "=========================================="
    echo "❌ Installation failed!"
    echo "=========================================="
    echo "Please check the error messages above"
    exit 1
fi