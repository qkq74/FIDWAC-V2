# FIDWAC v2 Installation Guide

This guide contains complete installation instructions for Windows (WSL2), Linux, and macOS, including Conda setup and troubleshooting.

## System Requirements

### Operating Systems

- Windows 10 or Windows 11 with WSL2 (Windows Subsystem for Linux 2)
- Linux (native):
  - Debian/Ubuntu 20.04 LTS or newer
  - Fedora 35+
  - RHEL 8+
  - Arch Linux
  - Other distributions with Python 3.9+
- macOS (Intel or Apple Silicon) with Python 3.10+ and Tk support

### Required Components

On Windows:
- WSL2 installed and configured
- Any Linux distribution (Debian recommended)

On Linux (WSL2 or native):
- Python 3.9 or newer
- pip (Python package manager)
- Basic build tools: build-essential or equivalent
- For full functionality (JPEG compression, GDAL, etc.):
  - libgdal-dev (Debian/Ubuntu), gdal-devel (Fedora/RHEL), gdal (Arch)
  - libjpeg62-turbo-dev (Debian/Ubuntu), libjpeg-turbo-devel (Fedora/RHEL), libjpeg-turbo (Arch)
  - llvm-dev (Debian/Ubuntu), llvm-devel (Fedora/RHEL), llvm (Arch)
  - p7zip-full (Debian/Ubuntu), p7zip (Fedora/RHEL/Arch)

On macOS:
- Python 3.10+ with tkinter support (Conda or python.org installer recommended)
- Homebrew packages for full functionality:
  - gdal
  - jpeg-turbo
  - p7zip
- Xcode Command Line Tools: xcode-select --install

RAM: minimum 2 GB, recommended 4+ GB for large rasters
Disk space: minimum 500 MB (+ space for processed raster files)

## Installation on Windows (WSL2)

### Step 1: Prepare WSL2

If you do not have WSL2 yet, install it:

```cmd
wsl --install
```

Restart your computer. After restart, open your Linux distribution from the Start menu and set up your username and password.

Verify WSL2 is working:

```cmd
wsl --list --verbose
```

You should see your distribution listed with version 2.

### Step 2: Install FIDWAC v2

Open Command Prompt and navigate to the project directory:

```cmd
cd path\to\FIDWAC_v2
```

Run the installation script:

```cmd
install\setup_windows.bat
```

The script will:
1. Check available WSL2 distributions
2. Ask you to select a distribution (or install a new one)
3. Ask you to select/create a Linux user
4. Install Python 3 and required system packages
5. Create a Python virtual environment at /home/user/.fidwac/venv
6. Install Python dependencies from install/requirements.txt

Expected time: 5-15 minutes (depends on internet speed and disk).

### Step 3: Run the Application

After successful installation:

```cmd
run_windows.bat
```

The FIDWAC v2 GUI should open in a Tkinter window.

## Installation on Linux (native)

### Step 1: Prepare Your System

On Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

On Fedora/RHEL:

```bash
sudo dnf install -y python3 python3-pip
```

On Arch:

```bash
sudo pacman -S python python-pip
```

### Step 2: Install FIDWAC v2

Navigate to the project directory:

```bash
cd /path/to/FIDWAC_v2
```

Run the installation script:

```bash
bash install/setup_linux.sh
```

The script will install:
1. A Python virtual environment at ~/.fidwac/venv
2. Required system packages (GDAL, libjpeg-turbo, LLVM, p7zip)
3. Python dependencies from install/requirements.txt

If system package installation fails, the script will display instructions on how to install them manually for your distribution.

### Step 3: Run the Application

```bash
bash run_linux.sh
```

The FIDWAC v2 GUI should open.

## Installation on macOS

### Option A: Virtual environment + Homebrew

1. Install Apple command-line tools:

```bash
xcode-select --install
```

2. Install system packages:

```bash
brew install gdal jpeg-turbo p7zip
```

3. Create and activate a virtual environment:

```bash
cd /path/to/FIDWAC_v2
python3 -m venv ~/.fidwac/venv
source ~/.fidwac/venv/bin/activate
python -m pip install --upgrade pip
pip install -r install/requirements.txt
```

4. Verify the environment:

```bash
python3 verify_install.py --require-gui --require-turbojpeg
```

5. Run the GUI or CLI:

```bash
python3 app.py
```

If python3 -c "import tkinter" fails, use a Python distribution that includes Tk support, such as Conda or the official python.org installer.

## Installation with Conda

The repository includes a ready-to-use Conda environment for macOS, Linux, and WSL2.

```bash
cd /path/to/FIDWAC_v2
conda env create -f environment.yml
conda activate fidwac
python3 verify_install.py --require-gui --require-turbojpeg
```

## Installation Troubleshooting

### Issue: "Virtual environment not found"

Cause: The run script could not find the installed virtual environment.

Solution:
1. Make sure you ran setup_windows.bat (Windows) or setup_linux.sh (Linux)
2. Wait for installation to complete
3. Try again

### Issue: Missing system packages (warning during installation)

Cause: Some system packages could not be installed automatically.

Solution: The script will display instructions on how to install them manually for your distribution. Follow those instructions and run the recommended sudo apt install, sudo dnf install, etc. commands.

### Issue: "PyTurboJPEG not available"

Cause: The libjpeg-turbo-dev library did not install properly.

Solution: JPEG installation is optional. The application will work, but JPEG compression via TurboJPEG will be unavailable. You can install it manually following the instructions provided by the installer, or ignore this warning.

### Issue: "No module named _tkinter"

Cause: Your Python build does not include Tk support.

Solution: Use the Conda environment from environment.yml, or install Python from python.org and rerun:

```bash
python3 verify_install.py --require-gui --require-turbojpeg
```
