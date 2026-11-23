#!/bin/bash
# =============================================================================
# VEGA (JARVIS OMEGA) - One-Click Setup Script
# =============================================================================
# This script sets up the complete VEGA environment:
# - System dependencies
# - Python environment with all packages
# - Model downloads (Whisper, YOLO, face recognition, etc.)
# - Configuration files
# - Docker images
# - Database initialization
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VEGA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VEGA_ROOT}/.venv"
MODELS_DIR="${VEGA_ROOT}/models"
DATA_DIR="${VEGA_ROOT}/data"
LOGS_DIR="${VEGA_ROOT}/logs"
CONFIG_FILE="${VEGA_ROOT}/.env"

# Default versions
PYTHON_VERSION="3.11"
NODE_VERSION="20"
DOCKER_COMPOSE_VERSION="2.24.0"

# Flags
SKIP_MODELS=false
SKIP_DOCKER=false
SKIP_DB=false
FORCE=false
DEV_MODE=false

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}=================================================================${NC}"
    echo ""
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

check_python_version() {
    if check_command python3; then
        local version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        local major=$(echo $version | cut -d. -f1)
        local minor=$(echo $version | cut -d. -f2)
        if [[ $major -ge 3 && $minor -ge 10 ]]; then
            log_success "Python $version found"
            return 0
        fi
    fi
    return 1
}

install_system_deps() {
    print_header "Installing System Dependencies"

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Detect distribution
        if check_command apt-get; then
            log_info "Detected Debian/Ubuntu"
            sudo apt-get update
            sudo apt-get install -y \
                python3 python3-venv python3-dev \
                build-essential cmake pkg-config \
                libopencv-dev libglib2.0-dev libsm6 libxext6 libxrender-dev \
                libgl1-mesa-glx portaudio19-dev \
                ffmpeg libsndfile1 \
                redis-server postgresql postgresql-contrib \
                curl wget git unzip \
                nvidia-container-toolkit || true
        elif check_command dnf; then
            log_info "Detected Fedora/RHEL"
            sudo dnf install -y \
                python3 python3-devel \
                gcc gcc-c++ make cmake pkgconfig \
                opencv-devel glib2-devel libSM libXext libXrender \
                mesa-libGL portaudio-devel \
                ffmpeg libsndfile \
                redis postgresql postgresql-server \
                curl wget git unzip
        elif check_command pacman; then
            log_info "Detected Arch Linux"
            sudo pacman -S --needed \
                python python-virtualenv \
                base-devel cmake pkgconf \
                opencv glib2 libsm libxext libxrender \
                mesa portaudio \
                ffmpeg libsndfile \
                redis postgresql \
                curl wget git unzip
        else
            log_warning "Unknown Linux distribution. Please install dependencies manually."
        fi

    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_info "Detected macOS"
        if check_command brew; then
            brew install \
                python@3.11 \
                cmake pkg-config \
                opencv glib portaudio \
                ffmpeg libsndfile \
                redis postgresql@15 \
                curl wget git
            # Start services
            brew services start redis
            brew services start postgresql@15
        else
            log_error "Homebrew not found. Please install Homebrew first: https://brew.sh"
            exit 1
        fi

    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        log_info "Detected Windows (Git Bash/WSL)"
        log_warning "For Windows, we recommend using WSL2 with Ubuntu."
        log_warning "Please run this script inside WSL2."
        exit 1
    fi

    log_success "System dependencies installed"
}

setup_python_env() {
    print_header "Setting Up Python Environment"

    # Create virtual environment
    if [[ -d "$VENV_DIR" && "$FORCE" != "true" ]]; then
        log_info "Virtual environment already exists at $VENV_DIR"
    else
        log_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
        log_success "Virtual environment created"
    fi

    # Activate and upgrade pip
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel

    # Install Python dependencies
    log_info "Installing Python packages..."
    pip install -r "${VEGA_ROOT}/requirements.txt"

    # Install PyTorch with CUDA support (if NVIDIA GPU)
    if check_command nvidia-smi; then
        log_info "NVIDIA GPU detected, installing PyTorch with CUDA..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log_info "No NVIDIA GPU detected, installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    log_success "Python environment ready"
}

download_models() {
    if [[ "$SKIP_MODELS" == "true" ]]; then
        log_warning "Skipping model downloads (--skip-models flag)"
        return
    fi

    print_header "Downloading AI Models"

    mkdir -p "$MODELS_DIR"
    cd "$MODELS_DIR"

    # Whisper models
    log_info "Downloading Whisper models..."
    python3 -c "
import whisper
for model in ['tiny', 'base', 'small', 'medium', 'large-v3']:
    try:
        whisper.load_model(model, download_root='./whisper')
        print(f'Downloaded {model}')
    except Exception as e:
        print(f'Failed to download {model}: {e}')
"

    # YOLOv8 models
    log_info "Downloading YOLOv8 models..."
    python3 -c "
from ultralytics import YOLO
for model in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']:
    try:
        YOLO(model)
        print(f'Downloaded {model}')
    except Exception as e:
        print(f'Failed to download {model}: {e}')
"

    # Face recognition models (DeepFace)
    log_info "Downloading face recognition models..."
    python3 -c "
from deepface import DeepFace
models = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace', 'GhostFaceNet']
for model in models:
    try:
        DeepFace.build_model(model)
        print(f'Downloaded {model}')
    except Exception as e:
        print(f'Failed to download {model}: {e}')
"

    # MediaPipe models (auto-downloaded on first use)
    log_info "MediaPipe models will auto-download on first use"

    # Piper TTS voices
    log_info "Downloading Piper TTS voices..."
    mkdir -p piper
    cd piper
    for voice in en_US-amy-medium en_US-ryan-medium en_US-lessac-medium; do
        if [[ ! -f "${voice}.onnx" ]]; then
            wget -q "https://github.com/rhasspy/piper/releases/download/v1.2.0/${voice}.onnx" || true
            wget -q "https://github.com/rhasspy/piper/releases/download/v1.2.0/${voice}.onnx.json" || true
        fi
    done

    log_success "Models downloaded"
}

setup_directories() {
    print_header "Setting Up Directories"

    mkdir -p "$DATA_DIR"
    mkdir -p "$LOGS_DIR"
    mkdir -p "${VEGA_ROOT}/.vega/workspace"
    mkdir -p "${VEGA_ROOT}/.vega/face_db"
    mkdir -p "${VEGA_ROOT}/.vega/plugins"
    mkdir -p "${VEGA_ROOT}/.vega/tools"
    mkdir -p "${VEGA_ROOT}/.vega/briefings"

    # Set permissions
    chmod 755 "$DATA_DIR" "$LOGS_DIR" "${VEGA_ROOT}/.vega"

    log_success "Directories created"
}

setup_config() {
    print_header "Setting Up Configuration"

    if [[ -f "$CONFIG_FILE" && "$FORCE" != "true" ]]; then
        log_info "Configuration file already exists at $CONFIG_FILE"
        log_info "Use --force to overwrite"
    else
        log_info "Creating configuration from template..."
        cp "${VEGA_ROOT}/.env.example" "$CONFIG_FILE"
        log_warning "Please edit $CONFIG_FILE and add your API keys:"
        log_warning "  - OPENAI_API_KEY"
        log_warning "  - ANTHROPIC_API_KEY"
        log_warning "  - DEEPGRAM_API_KEY"
        log_warning "  - ELEVENLABS_API_KEY"
        log_warning "  - HOME_ASSISTANT_TOKEN"
        log_warning "  - E2B_API_KEY"
    fi

    log_success "Configuration ready"
}

build_docker() {
    if [[ "$SKIP_DOCKER" == "true" ]]; then
        log_warning "Skipping Docker build (--skip-docker flag)"
        return
    fi

    print_header "Building Docker Images"

    cd "$VEGA_ROOT"

    # Build backend
    log_info "Building backend image..."
    docker build -t vega-backend:latest -f backend/Dockerfile backend/

    # Build frontend
    log_info "Building frontend image..."
    docker build -t vega-frontend:latest -f frontend/Dockerfile frontend/

    # Build vision services
    log_info "Building vision services..."
    docker build -t vega-yolo:latest -f docker/Dockerfile.yolo .
    docker build -t vega-face:latest -f docker/Dockerfile.face .
    docker build -t vega-gesture:latest -f docker/Dockerfile.gesture .
    docker build -t vega-screen-parser:latest -f docker/Dockerfile.screen_parser .

    log_success "Docker images built"
}

init_databases() {
    if [[ "$SKIP_DB" == "true" ]]; then
        log_warning "Skipping database initialization (--skip-db flag)"
        return
    fi

    print_header "Initializing Databases"

    # Wait for PostgreSQL
    log_info "Waiting for PostgreSQL..."
    for i in {1..30}; do
        if docker exec vega-postgres pg_isready -U vega -d vega &> /dev/null; then
            break
        fi
        sleep 2
    done

    # Run init script
    if [[ -f "${VEGA_ROOT}/scripts/init-db.sql" ]]; then
        log_info "Running database initialization..."
        docker exec -i vega-postgres psql -U vega -d vega < "${VEGA_ROOT}/scripts/init-db.sql"
    fi

    # Wait for Neo4j
    log_info "Waiting for Neo4j..."
    for i in {1..30}; do
        if docker exec vega-neo4j cypher-shell -u neo4j -p vega_graph_password "RETURN 1" &> /dev/null; then
            break
        fi
        sleep 2
    done

    # Create Neo4j indexes
    log_info "Creating Neo4j indexes..."
    docker exec vega-neo4j cypher-shell -u neo4j -p vega_graph_password "
        CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
        CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
        CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.id);
        CREATE INDEX conversation_id IF NOT EXISTS FOR (c:Conversation) ON (c.id);
    " || true

    log_success "Databases initialized"
}

run_tests() {
    print_header "Running Tests"

    source "$VENV_DIR/bin/activate"
    cd "$VEGA_ROOT"

    # Run pytest
    if [[ -f "pytest.ini" || -f "pyproject.toml" ]]; then
        pytest tests/ -v --tb=short
    else
        log_warning "No test configuration found, skipping tests"
    fi

    log_success "Tests completed"
}

create_systemd_service() {
    print_header "Creating Systemd Service (Linux only)"

    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        log_info "Not on Linux, skipping systemd service creation"
        return
    fi

    local service_file="/etc/systemd/system/vega.service"
    local user=$(whoami)

    if [[ -f "$service_file" && "$FORCE" != "true" ]]; then
        log_info "Systemd service already exists"
        return
    fi

    log_info "Creating systemd service at $service_file..."

    sudo tee "$service_file" > /dev/null <<EOF
[Unit]
Description=VEGA (JARVIS OMEGA) - Advanced AI Assistant
After=network.target docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$VEGA_ROOT
ExecStart=/usr/bin/docker-compose -f $VEGA_ROOT/docker-compose.yml up -d
ExecStop=/usr/bin/docker-compose -f $VEGA_ROOT/docker-compose.yml down
TimeoutStartSec=300
User=$user
Group=$(id -gn $user)

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable vega.service

    log_success "Systemd service created and enabled"
    log_info "Start with: sudo systemctl start vega"
    log_info "Stop with: sudo systemctl stop vega"
    log_info "Logs: sudo journalctl -u vega -f"
}

print_summary() {
    print_header "VEGA Setup Complete! 🎉"

    echo -e "${GREEN}Installation Summary:${NC}"
    echo "  • Python environment: $VENV_DIR"
    echo "  • Models directory: $MODELS_DIR"
    echo "  • Data directory: $DATA_DIR"
    echo "  • Logs directory: $LOGS_DIR"
    echo "  • Configuration: $CONFIG_FILE"
    echo ""

    echo -e "${BLUE}Next Steps:${NC}"
    echo "  1. Edit configuration: nano $CONFIG_FILE"
    echo "  2. Add your API keys (OpenAI, Anthropic, Deepgram, ElevenLabs, etc.)"
    echo "  3. Start services: docker-compose up -d"
    echo "  3. Or use systemd: sudo systemctl start vega"
    echo "  4. Access HUD: http://localhost:3000"
    echo "  5. Access API docs: http://localhost:8000/docs"
    echo "  6. Access Grafana: http://localhost:3001 (admin/vega_grafana_password)"
    echo ""

    echo -e "${BLUE}Useful Commands:${NC}"
    echo "  • View logs: docker-compose logs -f"
    echo "  • Stop services: docker-compose down"
    echo "  • Update VEGA: git pull && ./scripts/setup.sh"
    echo "  • Run tests: source .venv/bin/activate && pytest"
    echo ""

    echo -e "${YELLOW}Note:${NC} First run will download additional models on-demand."
    echo -e "${YELLOW}      GPU acceleration requires NVIDIA drivers + nvidia-container-toolkit.${NC}"
    echo ""
}

# =============================================================================
# MAIN
# =============================================================================

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --skip-models     Skip AI model downloads"
    echo "  --skip-docker     Skip Docker image builds"
    echo "  --skip-db         Skip database initialization"
    echo "  --force           Overwrite existing files"
    echo "  --dev             Development mode (additional tools)"
    echo "  --help            Show this help"
    echo ""
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-models) SKIP_MODELS=true ;;
            --skip-docker) SKIP_DOCKER=true ;;
            --skip-db) SKIP_DB=true ;;
            --force) FORCE=true ;;
            --dev) DEV_MODE=true ;;
            --help) usage; exit 0 ;;
            *) log_error "Unknown option: $1"; usage; exit 1 ;;
        esac
        shift
    done

    print_header "VEGA (JARVIS OMEGA) Setup"
    log_info "Setting up VEGA in $VEGA_ROOT"

    # Check prerequisites
    if ! check_python_version; then
        log_error "Python 3.10+ required. Please install Python 3.10 or higher."
        exit 1
    fi

    if ! check_command docker; then
        log_error "Docker not found. Please install Docker first."
        exit 1
    fi

    if ! check_command docker-compose; then
        log_error "Docker Compose not found. Please install Docker Compose v2."
        exit 1
    fi

    # Run setup steps
    install_system_deps
    setup_python_env
    download_models
    setup_directories
    setup_config
    build_docker
    init_databases
    run_tests
    create_systemd_service
    print_summary
}

main "$@"