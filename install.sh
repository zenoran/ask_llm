#!/usr/bin/env bash
#
# ask_llm installer
# 
# Usage (from GitHub):
#   curl -fsSL https://raw.githubusercontent.com/zenoran/ask_llm/master/install.sh | bash
#
# Or with options:
#   curl -fsSL https://raw.githubusercontent.com/zenoran/ask_llm/master/install.sh | bash -s -- --with-llama --with-hf
#
# Options:
#   --with-llama    Install llama-cpp-python for local GGUF models
#   --with-hf       Install HuggingFace transformers + torch
#   --all           Install all optional dependencies
#   --no-cuda       Skip CUDA support for llama-cpp-python
#   --uninstall     Remove ask_llm
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Defaults
INSTALL_LLAMA=false
INSTALL_HF=false
WITH_CUDA=true
UNINSTALL=false
REPO="git+https://github.com/zenoran/ask_llm.git"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-llama)
            INSTALL_LLAMA=true
            shift
            ;;
        --with-hf)
            INSTALL_HF=true
            shift
            ;;
        --all)
            INSTALL_LLAMA=true
            INSTALL_HF=true
            shift
            ;;
        --no-cuda)
            WITH_CUDA=false
            shift
            ;;
        --uninstall)
            UNINSTALL=true
            shift
            ;;
        --local)
            # For development: install from local path
            REPO="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╭─────────────────────────────╮${NC}"
echo -e "${BLUE}│   ask_llm Installer         │${NC}"
echo -e "${BLUE}╰─────────────────────────────╯${NC}"
echo

# Uninstall
if [ "$UNINSTALL" = true ]; then
    echo -e "${YELLOW}Uninstalling ask_llm...${NC}"
    pipx uninstall ask-llm 2>/dev/null || true
    echo -e "${GREEN}✓ ask_llm uninstalled${NC}"
    exit 0
fi

# Check for Python 3.12+
echo -e "${BLUE}Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.12+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
    echo -e "${RED}✗ Python 3.12+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Check/install pipx
echo -e "${BLUE}Checking pipx...${NC}"
if ! command -v pipx &> /dev/null; then
    echo -e "${YELLOW}Installing pipx...${NC}"
    python3 -m pip install --user pipx
    python3 -m pipx ensurepath
    # Source the updated PATH
    export PATH="$HOME/.local/bin:$PATH"
fi
echo -e "${GREEN}✓ pipx available${NC}"

# Install ask_llm
echo -e "${BLUE}Installing ask_llm...${NC}"
pipx install --force "$REPO"
echo -e "${GREEN}✓ ask_llm installed${NC}"

# Install optional dependencies
if [ "$INSTALL_LLAMA" = true ]; then
    echo -e "${BLUE}Installing llama-cpp-python...${NC}"
    
    if [ "$WITH_CUDA" = true ]; then
        # Check for CUDA
        if command -v nvcc &> /dev/null || [ -d "/usr/local/cuda" ]; then
            echo -e "${YELLOW}  CUDA detected, building with GPU support...${NC}"
            # Get CUDA version to determine supported architectures
            # CUDA 12.0 supports up to sm_90 (Hopper), not sm_120 (Blackwell)
            # Use explicit architectures to avoid "native" detection issues
            CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
            if [ -n "$CUDA_VERSION" ]; then
                echo -e "${YELLOW}  CUDA version: $CUDA_VERSION${NC}"
            fi
            # Build for common architectures (Pascal through Hopper)
            # sm_60=Pascal, sm_70=Volta, sm_75=Turing, sm_80=Ampere, sm_86=Ampere, sm_89=Ada, sm_90=Hopper
            CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=60;70;75;80;86;89;90" \
                pipx runpip ask-llm install llama-cpp-python --force-reinstall --no-cache-dir
        else
            echo -e "${YELLOW}  No CUDA detected, installing CPU-only version...${NC}"
            pipx runpip ask-llm install llama-cpp-python
        fi
    else
        echo -e "${YELLOW}  Installing CPU-only version (--no-cuda)...${NC}"
        pipx runpip ask-llm install llama-cpp-python
    fi
    echo -e "${GREEN}✓ llama-cpp-python installed${NC}"
fi

if [ "$INSTALL_HF" = true ]; then
    echo -e "${BLUE}Installing HuggingFace dependencies...${NC}"
    pipx runpip ask-llm install transformers torch huggingface-hub accelerate
    echo -e "${GREEN}✓ HuggingFace dependencies installed${NC}"
fi

# Verify installation
echo
echo -e "${BLUE}Verifying installation...${NC}"
if command -v llm &> /dev/null; then
    echo -e "${GREEN}✓ 'llm' command available${NC}"
else
    echo -e "${YELLOW}⚠ 'llm' not in PATH. You may need to restart your shell or run:${NC}"
    echo -e "  ${YELLOW}export PATH=\"\$HOME/.local/bin:\$PATH\"${NC}"
fi

echo
echo -e "${GREEN}╭─────────────────────────────╮${NC}"
echo -e "${GREEN}│   Installation Complete!    │${NC}"
echo -e "${GREEN}╰─────────────────────────────╯${NC}"
echo
echo -e "Commands available:"
echo -e "  ${BLUE}llm${NC}      - Query LLM models"
echo -e "  ${BLUE}ask-llm${NC}  - Same as llm"
echo
echo -e "Quick start:"
echo -e "  ${BLUE}llm --status${NC}           - Check system status"
echo -e "  ${BLUE}llm --list-models${NC}      - List available models"
echo -e "  ${BLUE}llm \"Hello, world!\"${NC}    - Ask a question"
echo
echo -e "Configuration: ${YELLOW}~/.config/ask-llm/.env${NC}"
echo
