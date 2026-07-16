#!/bin/bash
# FIDWAC v2 - Linux Run Script (for WSL2)
# Activates virtual environment and launches the application

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Use VENV_PATH from environment (set by run_windows.bat), or default location
if [ -n "$VENV_PATH" ]; then
    VENV_DIR="$VENV_PATH"
else
    VENV_DIR="$HOME/.fidwac/venv"
fi

cd "$PROJECT_ROOT"

# Set up LD_LIBRARY_PATH for libjpeg-turbo if compiled from source
LIBJPEG_PATH="/opt/libjpeg-turbo"
if [ -d "$LIBJPEG_PATH" ]; then
    export LD_LIBRARY_PATH="$LIBJPEG_PATH/lib64:$LIBJPEG_PATH/lib:$LD_LIBRARY_PATH"
fi

echo "========================================"
echo "Running FIDWAC v2"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "[ERROR] Virtual environment not found at: $VENV_DIR"
    echo ""
    echo "====================================================================="
    echo "WSL ENVIRONMENT ERROR"
    echo "====================================================================="
    echo "Virtual environment is missing or was corrupted."
    echo "To resolve this issue, please:"
    echo "  1. Delete the 'install/wsl_config.txt' file in Windows/Linux"
    echo "  2. Run the installer again: install/setup_windows.bat (from Windows)"
    echo "====================================================================="
    echo ""
    exit 1
fi

echo "[OK] Virtual environment detected: $VENV_DIR"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "[ERROR] app.py not found in: $PROJECT_ROOT"
    echo ""
    echo "Make sure you're running this script from the FIDWAC v2 project root directory."
    exit 1
fi

echo "[OK] app.py detected"
echo ""

# Check PyTurboJPEG version and libjpeg-turbo compatibility
echo "Checking PyTurboJPEG / libjpeg-turbo installation..."
python3 << 'PYEOF'
import sys
try:
    import turbojpeg
    print("[OK] PyTurboJPEG detected")
    
    # Try to get version info
    try:
        tj = turbojpeg.TurboJPEG()
        print("[OK] TurboJPEG instance created successfully")
    except RuntimeError as e:
        if "libjpeg-turbo 3.0" in str(e) or "3.0 or later" in str(e):
            print("[WARNING] libjpeg-turbo version mismatch detected!")
            print("[INFO] Falling back to PyTurboJPEG 1.x compatibility mode (libjpeg-turbo 2.x)")
            print("[INFO] Application will still work, but may use different code paths")
        elif "2.x" in str(e):
            print("[WARNING] libjpeg-turbo 2.x detected but newer version may be required")
            print("[INFO] Attempting to use compatibility layer...")
        else:
            print(f"[WARNING] TurboJPEG warning: {e}")
except ImportError:
    print("[WARNING] PyTurboJPEG not available")
    print("[INFO] JPEG quality mode will not be available")
    print("[INFO] Application will work in accuracy mode only")
    print("")
    print("To enable JPEG quality mode, install PyTurboJPEG:")
    print("  pip install PyTurboJPEG")
PYEOF
echo ""

# Launch the application
echo "Launching application..."
echo ""
python3 app.py
