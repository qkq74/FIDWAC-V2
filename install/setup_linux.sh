#!/bin/bash
# FIDWAC v2 - Linux Setup Script (WSL2 or native Linux)
# Sets up virtual environment and installs dependencies (no CUDA)

set +e  # Do not exit on error (we handle errors manually)

# Initialize variables
FAILED_DEPS=()
PKG_MANAGER=""
LIBJPEG_INSTALL_PATH=""
LIBJPEG_FOUND=0

echo "========================================"
echo "FIDWAC v2 Linux Installer"
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Project path: $PROJECT_ROOT"
echo ""

# Detect execution environment
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[ERROR] This installer supports only Linux and WSL2"
    exit 1
fi

if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "[OK] WSL2 environment detected"
else
    echo "[OK] Native Linux environment detected"
fi

echo ""

# Check if Python 3 is installed
echo "Checking Python 3..."
if ! command -v python3 &> /dev/null; then
    echo "[WARNING] Python 3 is not installed"
    echo "Installing Python 3 and related packages..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install Python 3"
        exit 1
    fi
    echo "[OK] Python 3 installed"
fi

PYTHON_VERSION=$(python3 --version)
echo "[OK] $PYTHON_VERSION"
echo ""

# Check if virtual environment exists
# Store venv in user's home directory (.fidwac), not in project directory
VENV_DIR="$HOME/.fidwac/venv"
if [ -d "$VENV_DIR" ]; then
    echo "[INFO] Virtual environment already exists: $VENV_DIR"
    echo "Do you want to recreate it? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment."
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    mkdir -p "$(dirname "$VENV_DIR")"
    python3 -m venv "$VENV_DIR"
    echo "[OK] Virtual environment created: $VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

echo ""

# Install system dependencies
echo "Checking system dependencies..."

# Detect package manager and distribution
PKG_MANAGER=""
DISTRO=""
FAILED_DEPS=()

if command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt"
    DISTRO="Debian/Ubuntu"
    GDAL_PKG="libgdal-dev"
    JPEG_PKG="libjpeg62-turbo-dev"
    LLVM_PKG="llvm-dev"
    P7ZIP_PKG="p7zip-full"
    CMAKE_PKG="cmake"
    BUILD_PKG="build-essential"
    PKG_CHECK_CMD="dpkg -l | grep -q"
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
    DISTRO="Fedora/RHEL 8+"
    GDAL_PKG="gdal-devel"
    JPEG_PKG="libjpeg-turbo-devel"
    LLVM_PKG="llvm-devel"
    P7ZIP_PKG="p7zip"
    CMAKE_PKG="cmake"
    BUILD_PKG="gcc-c++ make"
    PKG_CHECK_CMD="rpm -q"
elif command -v yum &> /dev/null; then
    PKG_MANAGER="yum"
    DISTRO="RHEL 7/CentOS"
    GDAL_PKG="gdal-devel"
    JPEG_PKG="libjpeg-turbo-devel"
    LLVM_PKG="llvm-devel"
    P7ZIP_PKG="p7zip"
    CMAKE_PKG="cmake"
    BUILD_PKG="gcc-c++ make"
    PKG_CHECK_CMD="rpm -q"
elif command -v pacman &> /dev/null; then
    PKG_MANAGER="pacman"
    DISTRO="Arch Linux"
    GDAL_PKG="gdal"
    JPEG_PKG="libjpeg-turbo"
    LLVM_PKG="llvm"
    P7ZIP_PKG="p7zip"
    CMAKE_PKG="cmake"
    BUILD_PKG="base-devel"
    PKG_CHECK_CMD="pacman -Q"
else
    echo "[WARNING] Package manager not detected. Skipping system dependencies."
    echo "Please install manually: GDAL, libjpeg-turbo, LLVM, p7zip"
    echo ""
fi

if [ -n "$PKG_MANAGER" ]; then
    echo "[INFO] Detected: $DISTRO (using $PKG_MANAGER)"
    echo ""
    
    MISSING_DEPS=()
    
    # Check packages
    if ! eval "$PKG_CHECK_CMD '$GDAL_PKG' > /dev/null 2>&1"; then
        MISSING_DEPS+=("$GDAL_PKG")
    fi
    if ! eval "$PKG_CHECK_CMD '$JPEG_PKG' > /dev/null 2>&1"; then
        MISSING_DEPS+=("$JPEG_PKG")
    fi
    if ! eval "$PKG_CHECK_CMD '$LLVM_PKG' > /dev/null 2>&1"; then
        MISSING_DEPS+=("$LLVM_PKG")
    fi
    if ! eval "$PKG_CHECK_CMD '$P7ZIP_PKG' > /dev/null 2>&1"; then
        MISSING_DEPS+=("$P7ZIP_PKG")
    fi
    
    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        echo "[WARNING] Missing system dependencies: ${MISSING_DEPS[*]}"
        echo "Attempting to install..."
        echo ""
        
        case "$PKG_MANAGER" in
            apt)
                sudo apt update -qq
                sudo apt install -y ${MISSING_DEPS[*]} 2>&1 | grep -E "^(Unable|E:|Package|Setting)" || true
                ;;
            dnf)
                sudo dnf install -y ${MISSING_DEPS[*]} 2>&1 | grep -E "^(Error|No package)" || true
                ;;
            yum)
                sudo yum install -y ${MISSING_DEPS[*]} 2>&1 | grep -E "^(Error|No package)" || true
                ;;
            pacman)
                sudo pacman -S --noconfirm ${MISSING_DEPS[*]} 2>&1 | grep -E "^(error|warning)" || true
                ;;
        esac
        
        # Check which packages failed to install
        for pkg in "${MISSING_DEPS[@]}"; do
            if ! eval "$PKG_CHECK_CMD '$pkg' > /dev/null 2>&1"; then
                FAILED_DEPS+=("$pkg")
            fi
        done
        
        if [ ${#FAILED_DEPS[@]} -eq 0 ]; then
            echo "[OK] System dependencies installed"
        else
            echo "[WARNING] Some packages could not be installed automatically"
            echo "Failed packages: ${FAILED_DEPS[*]}"
        fi
    else
        echo "[OK] System dependencies already installed"
    fi
    echo ""
fi

# =========================================================================
# libjpeg-turbo 3.0+ COMPILATION (if needed)
# =========================================================================
echo "Checking libjpeg-turbo version..."

TURBOJPEG_VERSION=$(python3 2>/dev/null << 'PYTHON_VERSION_CHECK'
import sys
try:
    from turbojpeg import TurboJPEG
    # Try to instantiate to check compatibility
    tj = TurboJPEG()
    print("3.0+")
    sys.exit(0)
except Exception:
    print("NOT_FOUND")
    sys.exit(1)
PYTHON_VERSION_CHECK
)

if [ "$TURBOJPEG_VERSION" != "3.0+" ]; then
    echo "[WARNING] libjpeg-turbo 3.0+ not found or incompatible"
    echo ""
    echo "Attempting to download and compile libjpeg-turbo 3.0+ from GitHub..."
    echo ""
    
    # First, ensure cmake and build tools are installed
    echo "Ensuring build tools are available..."
    MISSING_BUILD_TOOLS=()
    
    if ! command -v cmake &> /dev/null; then
        MISSING_BUILD_TOOLS+=("$CMAKE_PKG")
    fi
    if ! command -v make &> /dev/null; then
        MISSING_BUILD_TOOLS+=("$BUILD_PKG")
    fi
    
    if [ ${#MISSING_BUILD_TOOLS[@]} -gt 0 ]; then
        echo "Installing build tools: ${MISSING_BUILD_TOOLS[*]}"
        case "$PKG_MANAGER" in
            apt)
                sudo apt update -qq
                sudo apt install -y ${MISSING_BUILD_TOOLS[*]} > /dev/null 2>&1
                ;;
            dnf)
                sudo dnf install -y ${MISSING_BUILD_TOOLS[*]} > /dev/null 2>&1
                ;;
            yum)
                sudo yum install -y ${MISSING_BUILD_TOOLS[*]} > /dev/null 2>&1
                ;;
            pacman)
                sudo pacman -S --noconfirm ${MISSING_BUILD_TOOLS[*]} > /dev/null 2>&1
                ;;
        esac
    fi
    
    # Define install path
    LIBJPEG_INSTALL_PATH="/opt/libjpeg-turbo"
    LIBJPEG_BUILD_TMP="/tmp/libjpeg-turbo-build"
    
    # Check if it's already compiled there
    if [ -f "$LIBJPEG_INSTALL_PATH/lib/libturbojpeg.so.0" ] || [ -f "$LIBJPEG_INSTALL_PATH/lib64/libturbojpeg.so.0" ]; then
        echo "[INFO] Found pre-compiled libjpeg-turbo at $LIBJPEG_INSTALL_PATH"
        LIBJPEG_FOUND=1
    else
        # Clean old build if it exists
        rm -rf "$LIBJPEG_BUILD_TMP" 2>/dev/null || true
        mkdir -p "$LIBJPEG_BUILD_TMP"
        
        echo "Downloading latest libjpeg-turbo 3.x release..."
        cd "$LIBJPEG_BUILD_TMP"
        
        # Get the latest 3.x release URL
        LATEST_URL=$(python3 << 'PYTHON_GET_URL'
import urllib.request
import json
try:
    request = urllib.request.Request(
        "https://api.github.com/repos/libjpeg-turbo/libjpeg-turbo/releases",
        headers={"User-Agent": "FIDWAC-Installer"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        data = json.loads(response.read())
        for release in data:
            tag = release.get("tag_name", "")
            if tag.startswith("3."):
                # Find the source tarball
                for asset in release.get("assets", []):
                    if asset["name"].endswith(".tar.gz") and "windows" not in asset["name"].lower():
                        print(asset["browser_download_url"])
                        exit(0)
        print("ERROR")
except Exception as e:
    print("ERROR")
PYTHON_GET_URL
        )
        
        if [ "$LATEST_URL" = "ERROR" ] || [ -z "$LATEST_URL" ]; then
            echo "[ERROR] Could not determine latest libjpeg-turbo 3.x release"
            echo "Please install manually: sudo apt install libjpeg-turbo-dev (and ensure version 3.0+)"
            exit 1
        fi
        
        echo "URL: $LATEST_URL"
        
        # Try wget first, then curl
        if command -v wget &> /dev/null; then
            if ! wget -q "$LATEST_URL" -O libjpeg-turbo.tar.gz; then
                echo "[ERROR] Failed to download libjpeg-turbo"
                exit 1
            fi
        elif command -v curl &> /dev/null; then
            if ! curl -sL "$LATEST_URL" -o libjpeg-turbo.tar.gz; then
                echo "[ERROR] Failed to download libjpeg-turbo"
                exit 1
            fi
        else
            echo "[ERROR] Neither wget nor curl found. Cannot download libjpeg-turbo"
            exit 1
        fi
        
        echo "Extracting..."
        tar xzf libjpeg-turbo.tar.gz
        cd libjpeg-turbo-* 2>/dev/null || { echo "[ERROR] Failed to extract"; exit 1; }
        
        echo "Configuring and building (this may take a few minutes)..."
        mkdir build && cd build
        
        if ! cmake .. -DCMAKE_INSTALL_PREFIX="$LIBJPEG_INSTALL_PATH" -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1; then
            echo "[ERROR] CMake configuration failed"
            echo "Make sure cmake and build tools are installed"
            exit 1
        fi
        
        if ! make -j$(nproc) > /dev/null 2>&1; then
            echo "[ERROR] Build failed"
            exit 1
        fi
        
        echo "Installing to $LIBJPEG_INSTALL_PATH..."
        sudo make install > /dev/null 2>&1
        
        if [ $? -ne 0 ]; then
            echo "[ERROR] Installation failed (sudo required)"
            exit 1
        fi
        
        echo "[OK] libjpeg-turbo compiled and installed"
        LIBJPEG_FOUND=1
    fi
    
    cd "$PROJECT_ROOT"
    
    # Set environment variables for PyTurboJPEG
    export LD_LIBRARY_PATH="$LIBJPEG_INSTALL_PATH/lib64:$LIBJPEG_INSTALL_PATH/lib:$LD_LIBRARY_PATH"
    echo ""
    echo "Updated LD_LIBRARY_PATH for this session"
    echo ""
fi

# Critical section: require successful completion
set -e

# Install Python dependencies
echo "Installing Python dependencies..."
if [ -f "install/requirements.txt" ]; then
    pip install -r install/requirements.txt -q
    echo "[OK] Python dependencies installed"
else
    echo "[ERROR] install/requirements.txt not found!"
    exit 1
fi

echo ""

# Allow non-fatal errors again
set +e

# Verify installation
echo "Verifying installation..."
python3 -c "import numpy, scipy, rasterio, msgpack, py7zr, tqdm, numba; print('[OK] Core dependencies')" 
if [ $? -ne 0 ]; then
    echo "[ERROR] Core dependencies verification failed!"
    exit 1
fi

# Check PyTurboJPEG compatibility (REQUIRED)
echo ""
echo "Checking PyTurboJPEG/libjpeg-turbo compatibility..."
python3 << 'PYTHON_CHECK'
import sys
import os

try:
    # Try to import turbojpeg
    from turbojpeg import TurboJPEG
    print("[OK] PyTurboJPEG 2.0+ with libjpeg-turbo 3.0+ is available")
    sys.exit(0)
except OSError as e:
    error_msg = str(e)
    if "libjpeg-turbo" in error_msg.lower() and "3.0" in error_msg.lower():
        print("[ERROR] PyTurboJPEG requires libjpeg-turbo 3.0 or later")
        print(f"Error details: {error_msg}")
        print("")
        print("Your current libjpeg-turbo version is too old.")
        sys.exit(1)
    elif "libjpeg" in error_msg.lower():
        print("[ERROR] libjpeg-turbo not found or incompatible")
        print(f"Error details: {error_msg}")
        sys.exit(1)
    else:
        print(f"[ERROR] PyTurboJPEG initialization failed: {error_msg}")
        sys.exit(1)
except ImportError as e:
    print(f"[ERROR] PyTurboJPEG not installed: {e}")
    sys.exit(1)
PYTHON_CHECK

TURBOJPEG_CHECK=$?
if [ $TURBOJPEG_CHECK -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "ACTION REQUIRED: libjpeg-turbo 3.0+"
    echo "========================================"
    echo ""
    echo "PyTurboJPEG requires libjpeg-turbo 3.0 or later for SIMD encoding."
    echo "This is MANDATORY for efficient compression of large files (>2GB)."
    echo ""
    echo "To install/upgrade libjpeg-turbo 3.0+:"
    echo ""
    case "$PKG_MANAGER" in
        apt)
            echo "Debian/Ubuntu:"
            echo "  sudo apt remove libjpeg62-turbo-dev libjpeg-turbo 2>/dev/null || true"
            echo "  sudo apt update"
            echo "  sudo apt install libjpeg-turbo-dev"
            ;;
        dnf)
            echo "Fedora/RHEL 8+:"
            echo "  sudo dnf remove libjpeg-turbo-devel 2>/dev/null || true"
            echo "  sudo dnf install libjpeg-turbo-devel"
            ;;
        yum)
            echo "RHEL 7/CentOS:"
            echo "  sudo yum remove libjpeg-turbo-devel 2>/dev/null || true"
            echo "  sudo yum install libjpeg-turbo-devel"
            ;;
        pacman)
            echo "Arch Linux:"
            echo "  sudo pacman -R libjpeg-turbo 2>/dev/null || true"
            echo "  sudo pacman -S libjpeg-turbo"
            ;;
        *)
            echo "Your distribution:"
            echo "  Install or upgrade 'libjpeg-turbo' development package to 3.0+"
            ;;
    esac
    echo ""
    echo "After installation, verify with:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  python3 -c 'from turbojpeg import TurboJPEG; print(\"OK\")'"
    echo ""
    exit 1
fi

echo ""
echo "========================================"
echo "INSTALLATION COMPLETED"
echo "========================================"
echo ""

# Display any missing dependencies
if [ ${#FAILED_DEPS[@]} -gt 0 ]; then
    echo "[⚠️  WARNING] Some system dependencies could not be installed automatically:"
    echo ""
    for pkg in "${FAILED_DEPS[@]}"; do
        echo "  • $pkg"
    done
    echo ""
    echo "To install them manually, use your package manager:"
    case "$PKG_MANAGER" in
        apt)
            echo "  sudo apt install ${FAILED_DEPS[*]}"
            ;;
        dnf)
            echo "  sudo dnf install ${FAILED_DEPS[*]}"
            ;;
        yum)
            echo "  sudo yum install ${FAILED_DEPS[*]}"
            ;;
        pacman)
            echo "  sudo pacman -S ${FAILED_DEPS[*]}"
            ;;
        *)
            echo "  Install these packages using your distribution's package manager"
            ;;
    esac
    echo ""
    echo "The application may work without them, but some features might be limited."
    echo ""
fi

echo ""
echo "To run the application:"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 app.py"
echo ""
echo "Or use the script: run_linux.sh or in windows run_windows.bat"
echo ""

# Display libjpeg-turbo info if it was compiled
if [ "$LIBJPEG_FOUND" = "1" ] && [ -n "$LIBJPEG_INSTALL_PATH" ]; then
    echo "========================================"
    echo "libjpeg-turbo Information"
    echo "========================================"
    echo ""
    echo "libjpeg-turbo 3.0+ has been compiled and installed to:"
    echo "  $LIBJPEG_INSTALL_PATH"
    echo ""
    echo "The run_linux.sh script will automatically set LD_LIBRARY_PATH."
    echo "If you run the application manually, set it with:"
    echo "  export LD_LIBRARY_PATH=$LIBJPEG_INSTALL_PATH/lib64:$LIBJPEG_INSTALL_PATH/lib:\$LD_LIBRARY_PATH"
    echo ""
fi
