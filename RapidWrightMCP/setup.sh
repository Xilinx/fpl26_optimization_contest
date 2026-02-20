#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache 2.0

# RapidWright MCP Server Setup Script

set -e

# Allow user to specify Python executable via PYTHON_EXE env var (default: python3)
PYTHON_EXE="${PYTHON_EXE:-python3}"

echo "=== RapidWright MCP Server Setup ==="
echo ""

# Check Python version
echo "Checking Python version..."
echo "Using Python executable: $PYTHON_EXE"
if ! command -v "$PYTHON_EXE" &> /dev/null; then
    echo "ERROR: $PYTHON_EXE not found!"
    echo "Please install Python 3.8 or later, or set PYTHON_EXE to a valid Python executable"
    exit 1
fi

python_version=$("$PYTHON_EXE" --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Function to extract Java major version number
get_java_major_version() {
    local java_cmd="$1"
    local version_output=$("$java_cmd" -version 2>&1 | head -n 1)
    # Extract version number - handles both "1.8.0_xxx" and "11.0.x" formats
    local version=$(echo "$version_output" | sed -n 's/.*version "\([0-9]*\)\.\{0,1\}\([0-9]*\).*/\1/p')
    # For Java 1.x versions, the major version is the second number
    if [ "$version" = "1" ]; then
        version=$(echo "$version_output" | sed -n 's/.*version "1\.\([0-9]*\).*/\1/p')
    fi
    echo "$version"
}

# Function to find Java from Vivado installation
find_java_from_vivado() {
    echo "Looking for Vivado installation to extract Java..."
    
    if ! command -v vivado &> /dev/null; then
        echo "  Vivado not found in PATH"
        return 1
    fi
    
    # Get Vivado executable path
    local vivado_path=$(which vivado)
    echo "  Found Vivado at: $vivado_path"
    
    # Resolve symlinks to get the actual Vivado installation directory
    local vivado_real=$(readlink -f "$vivado_path" 2>/dev/null || echo "$vivado_path")
    
    # Extract the Vivado installation root (e.g., /opt/Vivado/2022.2)
    # The vivado binary is typically at <install>/bin/vivado
    local vivado_root=$(dirname $(dirname "$vivado_real"))
    echo "  Vivado installation root: $vivado_root"
    
    # Look for Java in the tps directory
    # Java is typically at <vivado_root>/tps/lnx64/jre*/bin/java
    local java_dir=""
    for jre_dir in "$vivado_root"/tps/lnx64/jre*; do
        if [ -d "$jre_dir" ] && [ -x "$jre_dir/bin/java" ]; then
            java_dir="$jre_dir"
            break
        fi
    done
    
    if [ -z "$java_dir" ]; then
        echo "  Could not find Java in Vivado installation"
        return 1
    fi
    
    echo "  Found Java at: $java_dir"
    
    # Verify Java version is 11+
    local java_version=$(get_java_major_version "$java_dir/bin/java")
    if [ -z "$java_version" ] || [ "$java_version" -lt 11 ]; then
        echo "  Warning: Java found in Vivado is version $java_version (need 11+)"
        return 1
    fi
    
    # Export the variables
    export JAVA_HOME="$java_dir"
    export PATH="$java_dir/bin:$PATH"
    
    echo "  ✓ Using Java $java_version from Vivado"
    echo "  Set JAVA_HOME=$JAVA_HOME"
    return 0
}

# Function to determine JAVA_HOME from java executable path
get_java_home() {
    local java_cmd="$1"
    local java_real=$(readlink -f "$(which "$java_cmd")" 2>/dev/null || which "$java_cmd")
    # Java is typically at $JAVA_HOME/bin/java, so go up two directories
    local java_home=$(dirname $(dirname "$java_real"))
    echo "$java_home"
}

# Check Java version
echo ""
echo "Checking Java version..."
java_ok=false

if command -v java &> /dev/null; then
    java_version=$(java -version 2>&1 | head -n 1)
    java_major=$(get_java_major_version "java")
    
    if [ -n "$java_major" ] && [ "$java_major" -ge 11 ]; then
        echo "✓ Java found: $java_version (version $java_major)"
        java_ok=true
        
        # Set JAVA_HOME from the found Java (JPype needs this)
        export JAVA_HOME=$(get_java_home "java")
        echo "  Set JAVA_HOME=$JAVA_HOME"
    else
        echo "⚠ Java found but version is too old: $java_version"
        echo "  RapidWright requires Java 11 or later"
    fi
else
    echo "⚠ Java not found in PATH"
fi

# If Java is missing or too old, try to find it from Vivado
if [ "$java_ok" = false ]; then
    echo ""
    if find_java_from_vivado; then
        java_ok=true
        echo ""
        echo "IMPORTANT: To persist Java settings, add these to your shell profile:"
        echo "  export JAVA_HOME=$JAVA_HOME"
        echo "  export PATH=\$JAVA_HOME/bin:\$PATH"
    else
        echo ""
        echo "ERROR: Could not find suitable Java (11+)"
        echo ""
        echo "Options to resolve:"
        echo "  1. Install Java 11 or later (e.g., from https://adoptium.net/)"
        echo "  2. Add Vivado to your PATH before running this script"
        echo "  3. Manually set JAVA_HOME to a Java 11+ installation"
        echo ""
        echo "If you have Vivado installed, you can use its bundled Java:"
        echo "  export JAVA_HOME=/path/to/Vivado/20XX.X/tps/lnx64/jreXX.X.XX_X"
        echo "  export PATH=\$JAVA_HOME/bin:\$PATH"
        exit 1
    fi
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
echo "This will install: mcp, rapidwright (includes JPype and RapidWright Java libraries)"
"$PYTHON_EXE" -m pip install -r requirements.txt

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "✓ Python dependencies installed"
echo "✓ RapidWright installed from pip (includes Java libraries)"
echo ""
echo "To test the server:"
echo "  $PYTHON_EXE test_server.py"
echo ""
echo "To start the server:"
echo "  $PYTHON_EXE server.py"
echo ""
echo "To use with Cursor/Claude Desktop, add to your MCP config:"
echo ""

# Always include JAVA_HOME in the MCP config (JPype needs this)
echo '{
  "mcpServers": {
    "rapidwright": {
      "command": "'$PYTHON_EXE'",
      "args": ["'$(pwd)'/server.py"],
      "env": {
        "JAVA_HOME": "'$JAVA_HOME'"
      }
    }
  }
}'

echo ""
echo "Note: No RAPIDWRIGHT_PATH needed - everything is managed by pip!"
echo ""

