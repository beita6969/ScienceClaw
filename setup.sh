#!/usr/bin/env bash
# ============================================================================
#  ScienceClaw — One-Click Setup
#  Make Research Easier Across All Disciplines
# ============================================================================
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'
BOLD='\033[1m'

print_banner() {
    echo -e "${CYAN}"
    echo "  ╔══════════════════════════════════════════════════════════╗"
    echo "  ║                                                          ║"
    echo "  ║   🦞 ScienceClaw — AI Research Gateway                   ║"
    echo "  ║   Make Research Easier Across All Disciplines            ║"
    echo "  ║                                                          ║"
    echo "  ╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log_step() { echo -e "\n${BLUE}[$(date +%H:%M:%S)]${NC} ${BOLD}▶ $1${NC}"; }
log_ok()   { echo -e "  ${GREEN}✅ $1${NC}"; }
log_warn() { echo -e "  ${YELLOW}⚠️  $1${NC}"; }
log_err()  { echo -e "  ${RED}❌ $1${NC}"; }
log_info() { echo -e "  ${CYAN}ℹ️  $1${NC}"; }

# ---- Detect OS ----
detect_os() {
    case "$(uname -s)" in
        Darwin*) OS="macos" ;;
        Linux*)  OS="linux" ;;
        *)       OS="unknown" ;;
    esac
    echo "$OS"
}

# ---- Check and install prerequisites ----
check_command() {
    if command -v "$1" &>/dev/null; then
        log_ok "$1 found: $(command -v "$1")"
        return 0
    else
        return 1
    fi
}

install_brew() {
    if ! check_command brew; then
        log_step "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
}

install_node() {
    if check_command node; then
        NODE_VER=$(node -v | sed 's/v//' | cut -d. -f1)
        if [ "$NODE_VER" -ge 20 ]; then
            log_ok "Node.js v$(node -v) (>= 20 required)"
            return
        else
            log_warn "Node.js $(node -v) is too old, need >= 20"
        fi
    fi
    log_step "Installing Node.js 20+..."
    if [ "$OS" = "macos" ]; then
        brew install node@22
    else
        curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi
}

install_pnpm() {
    if ! check_command pnpm; then
        log_step "Installing pnpm..."
        npm install -g pnpm
    fi
}

install_python() {
    if check_command python3; then
        PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        log_ok "Python $PY_VER found"
    else
        log_step "Installing Python 3..."
        if [ "$OS" = "macos" ]; then
            brew install python@3.12
        else
            sudo apt-get install -y python3 python3-pip python3-venv
        fi
    fi
}

install_uv() {
    if ! check_command uv; then
        log_step "Installing uv (Python package manager)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
}

# ---- Install Node.js dependencies ----
install_node_deps() {
    log_step "Installing Node.js dependencies..."
    pnpm install --no-frozen-lockfile 2>&1 | tail -5
    log_ok "Node.js dependencies installed"
}

# ---- Install Python scientific packages ----
install_python_deps() {
    log_step "Installing Python scientific packages..."

    # Core scientific stack
    CORE_PACKAGES=(
        numpy scipy statsmodels scikit-learn matplotlib pandas
        sympy networkx plotly seaborn polars pyarrow
    )

    # Bioinformatics & chemistry
    BIO_PACKAGES=(
        biopython pubchempy chembl_webresource_client
        rcsb-api rcsbsearchapi reactome2py
    )

    # NLP & text
    NLP_PACKAGES=(
        jieba sentence-transformers
    )

    # Document processing
    DOC_PACKAGES=(
        pdfplumber pymupdf pylatexenc Pillow
        defusedxml python-dotenv requests
    )

    # Data & visualization
    VIZ_PACKAGES=(
        kaleido dash folium cartopy
        geopandas mapclassify contextily
    )

    # Statistics & ML
    STAT_PACKAGES=(
        lifelines shap scikit-bio semopy
    )

    # Research tools
    RESEARCH_PACKAGES=(
        scholarly crossref-commons bibtexparser
        asreview bertopic litellm openai
        Wikipedia-API GEOparse cdsapi
    )

    ALL_PACKAGES=("${CORE_PACKAGES[@]}" "${BIO_PACKAGES[@]}" "${NLP_PACKAGES[@]}" \
                  "${DOC_PACKAGES[@]}" "${VIZ_PACKAGES[@]}" "${STAT_PACKAGES[@]}" \
                  "${RESEARCH_PACKAGES[@]}")

    log_info "Installing ${#ALL_PACKAGES[@]} Python packages..."

    # Create a virtual environment for ScienceClaw's Python deps
    VENV_DIR="$HOME/.scienceclaw/venv"
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating Python virtual environment at $VENV_DIR..."
        mkdir -p "$HOME/.scienceclaw"
        uv venv "$VENV_DIR" 2>&1 | tail -1
    fi

    log_info "Installing packages into virtual environment..."
    uv pip install --python "$VENV_DIR/bin/python" "${ALL_PACKAGES[@]}" 2>&1 | tail -5 || {
        log_warn "Some packages failed, retrying individually..."
        for pkg in "${ALL_PACKAGES[@]}"; do
            uv pip install --python "$VENV_DIR/bin/python" "$pkg" 2>/dev/null || log_warn "  $pkg failed (optional)"
        done
    }

    log_info "To activate: source $VENV_DIR/bin/activate"

    log_ok "Python scientific packages installed"
}

# ---- Install MCP servers ----
install_mcp_servers() {
    log_step "Installing MCP servers for academic research..."

    MCP_SERVERS=(
        "academic-mcp"
        "arxiv-mcp-server"
        "biomcp"
        "semantic-scholar-mcp"
        "zotero-mcp"
        "deep-research-mcp-server"
    )

    # MCP servers run via uvx on-demand (no pre-install needed)
    # We just verify uvx can resolve them
    for server in "${MCP_SERVERS[@]}"; do
        log_info "Caching $server..."
        uvx --quiet "$server" --help &>/dev/null && log_ok "$server cached" || log_warn "$server not available (will try at runtime)"
    done

    # Bundled MCP servers (not on PyPI, included in project)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    VENV_DIR="$HOME/.scienceclaw/venv"

    if [ -d "$SCRIPT_DIR/mcp-servers/chembl-mcp" ]; then
        log_info "Installing ChEMBL MCP dependencies..."
        uv pip install --python "$VENV_DIR/bin/python" -r "$SCRIPT_DIR/mcp-servers/chembl-mcp/requirements.txt" 2>&1 | tail -3 || true
        log_ok "ChEMBL MCP server ready (bundled)"
    fi

    if [ -d "$SCRIPT_DIR/mcp-servers/arxiv-latex-mcp" ]; then
        log_info "Installing arXiv LaTeX MCP dependencies..."
        uv pip install --python "$VENV_DIR/bin/python" httpx "mcp[cli]" "arxiv-to-prompt>=0.10.0" 2>&1 | tail -3 || true
        log_ok "arXiv LaTeX MCP server ready (bundled)"
    fi

    log_ok "MCP servers installed"
}

# ---- Configure ScienceClaw ----
configure_scienceclaw() {
    log_step "Configuring ScienceClaw..."

    CONFIG_DIR="$HOME/.openclaw"
    CONFIG_FILE="$CONFIG_DIR/openclaw.json"

    mkdir -p "$CONFIG_DIR"

    if [ -f "$CONFIG_FILE" ]; then
        log_info "Existing config found, creating backup..."
        cp "$CONFIG_FILE" "$CONFIG_FILE.bak.$(date +%Y%m%d%H%M%S)"
    fi

    VENV_PYTHON="$HOME/.scienceclaw/venv/bin/python"
    CHEMBL_SERVER="$HOME/clawd/mcp-servers/chembl-mcp/chembl_server.py"

    cat > "$CONFIG_FILE" << JSONEOF
{
  "gateway": {
    "mode": "local"
  },
  "plugins": {
    "slots": {
      "memory": "memory-core"
    },
    "entries": {
      "memory-core": { "enabled": true },
      "memory-lancedb": { "enabled": true }
    }
  },
  "agents": {
    "defaults": {
      "heartbeat": { "interval": 1800 }
    }
  }
}
JSONEOF

    log_ok "Config written to $CONFIG_FILE"

    # Run doctor to auto-fix any config issues
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    log_info "Running config validation..."
    cd "$SCRIPT_DIR" && npx openclaw doctor --fix 2>&1 | tail -3 || true
    log_ok "Config validated"
}

# ---- Setup workspace ----
setup_workspace() {
    log_step "Setting up research workspace..."

    WORKSPACE="$HOME/clawd"
    mkdir -p "$WORKSPACE/skills" "$WORKSPACE/output" "$WORKSPACE/data"

    # Copy skills to workspace
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -d "$SCRIPT_DIR/skills" ]; then
        log_info "Copying skills to workspace..."
        cp -rn "$SCRIPT_DIR/skills/"* "$WORKSPACE/skills/" 2>/dev/null || true
        SKILL_COUNT=$(ls -1 "$WORKSPACE/skills/" 2>/dev/null | wc -l | tr -d ' ')
        log_ok "$SKILL_COUNT skills available in workspace"
    fi

    # Copy bundled MCP servers to workspace
    if [ -d "$SCRIPT_DIR/mcp-servers" ]; then
        log_info "Copying bundled MCP servers..."
        mkdir -p "$WORKSPACE/mcp-servers"
        cp -rn "$SCRIPT_DIR/mcp-servers/"* "$WORKSPACE/mcp-servers/" 2>/dev/null || true
        log_ok "Bundled MCP servers copied"
    fi

    log_ok "Workspace ready at $WORKSPACE"
}

# ---- Main ----
main() {
    print_banner

    OS=$(detect_os)
    log_info "Detected OS: $OS"

    if [ "$OS" = "unknown" ]; then
        log_err "Unsupported OS. ScienceClaw supports macOS and Linux."
        log_info "For Windows, use WSL2: https://docs.microsoft.com/en-us/windows/wsl/"
        exit 1
    fi

    # Phase 1: Prerequisites
    log_step "Phase 1/5: Checking prerequisites..."
    [ "$OS" = "macos" ] && install_brew
    install_node
    install_pnpm
    install_python
    install_uv

    # Phase 2: Node.js
    log_step "Phase 2/5: Installing Node.js packages..."
    install_node_deps

    # Phase 3: Python
    log_step "Phase 3/5: Installing Python scientific stack..."
    install_python_deps

    # Phase 4: MCP Servers
    log_step "Phase 4/5: Installing MCP servers..."
    install_mcp_servers

    # Phase 5: Configure & Launch
    log_step "Phase 5/5: Configuring ScienceClaw..."
    configure_scienceclaw
    setup_workspace

    # Done
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "  ╔══════════════════════════════════════════════════════════╗"
    echo "  ║                                                          ║"
    echo "  ║   🦞 ScienceClaw setup complete!                         ║"
    echo "  ║                                                          ║"
    echo "  ║   To start:                                              ║"
    echo "  ║     npx openclaw onboard    (first time)                 ║"
    echo "  ║     npx openclaw gateway    (start gateway)              ║"
    echo "  ║                                                          ║"
    echo "  ║   Workspace: ~/clawd                                     ║"
    echo "  ║   Config:    ~/.openclaw/openclaw.json                   ║"
    echo "  ║                                                          ║"
    echo "  ╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

main "$@"
