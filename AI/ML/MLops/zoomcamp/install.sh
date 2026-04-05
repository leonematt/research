#!/bin/bash
set -e

echo "========================================"
echo "Starting ML Zoomcamp Environment Setup"
echo "========================================"

# 1. Update system packages
echo "[1/9] Checking system packages (curl, wget, git, bzip2, openssl)..."
sudo apt-get update -y -qq
sudo apt-get install -y curl wget git bzip2 openssl -qq

# 2. Install Docker
echo "[2/9] Checking Docker..."
if command -v docker &> /dev/null; then
    echo "✅ Docker is already installed: $(docker --version)"
else
    echo "⬇️ Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# 3. Install Docker Compose
echo "[3/9] Checking Docker Compose..."
if docker compose version &> /dev/null || command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose is already installed."
else
    echo "⬇️ Installing Docker Compose..."
    sudo apt-get install -y docker-compose-plugin
    sudo curl -SL "https://github.com/docker/compose/releases/download/v2.26.1/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# 4. Install Miniconda
echo "[4/9] Checking Miniconda..."
if [ -d "$HOME/miniconda" ] || command -v conda &> /dev/null; then
    echo "✅ Conda is already installed."
else
    echo "⬇️ Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
fi

# 5. Initialize Conda
echo "[5/9] Configuring Conda for Bash and Zsh..."
# Source activate specifically so this running script can use the conda command below
source $HOME/miniconda/bin/activate
conda init bash > /dev/null
conda init zsh > /dev/null

# 6. Accept Anaconda Terms of Service
echo "[6/9] Accepting Conda Terms of Service..."
# The '|| true' ensures the script doesn't fail if TOS was already accepted
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# 7. Create Conda Virtual Environment
echo "[7/9] Checking for ml-zoomcamp conda environment..."
if conda env list | grep -q "ml-zoomcamp"; then
    echo "✅ Conda environment 'ml-zoomcamp' already exists. Updating packages..."
    conda run -n ml-zoomcamp pip install -r requirements.txt
else
    echo "⬇️ Creating ml-zoomcamp conda environment..."
    conda create -y -n ml-zoomcamp python=3.11
    echo "⬇️ Installing Python requirements..."
    conda run -n ml-zoomcamp pip install -r requirements.txt
fi

# 8. Generate Self-Signed Certificate
echo "[8/9] Generating self-signed SSL certificate..."
CERT_DIR="$HOME/.jupyter/certs"
mkdir -p "$CERT_DIR"

if [ -f "$CERT_DIR/mycert.pem" ] && [ -f "$CERT_DIR/mykey.key" ]; then
    echo "✅ SSL certificates already exist in $CERT_DIR"
else
    # The -subj flag bypasses the interactive prompts during certificate creation
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$CERT_DIR/mykey.key" \
        -out "$CERT_DIR/mycert.pem" \
        -subj "/C=US/ST=Oregon/L=Hillsboro/O=MLZoomcamp/OU=Dev/CN=0.0.0.0" 2>/dev/null
    echo "✅ Created self-signed certificate in $CERT_DIR"
fi

# 9. Configure Jupyter Notebook
echo "[9/9] Configuring Jupyter Notebook for HTTPS on 0.0.0.0..."
JUPYTER_CONFIG_DIR="$HOME/.jupyter"
JUPYTER_CONFIG_FILE="$JUPYTER_CONFIG_DIR/jupyter_notebook_config.py"

# Generate default config if it doesn't exist
conda run -n ml-zoomcamp jupyter notebook --generate-config -y > /dev/null 2>&1 || true

# Append SSL and IP settings if not already present
if ! grep -q "ML Zoomcamp Auto-Config" "$JUPYTER_CONFIG_FILE" 2>/dev/null; then
    cat <<EOF >> "$JUPYTER_CONFIG_FILE"

# --- ML Zoomcamp Auto-Config ---
# Configuration for Jupyter Server (JupyterLab / Notebook v7+)
c.ServerApp.certfile = u'$CERT_DIR/mycert.pem'
c.ServerApp.keyfile = u'$CERT_DIR/mykey.key'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.open_browser = False

# Configuration for older Jupyter Notebooks (v6 and below)
c.NotebookApp.certfile = u'$CERT_DIR/mycert.pem'
c.NotebookApp.keyfile = u'$CERT_DIR/mykey.key'
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False
# -------------------------------
EOF
    echo "✅ Appended HTTPS and 0.0.0.0 settings to $JUPYTER_CONFIG_FILE"
else
    echo "✅ Jupyter HTTPS configuration already exists."
fi

echo "=========================================================="
echo "🎉 Installation & Verification Complete!"
echo "1. Run 'source ~/.zshrc' (or close/re-open your terminal)."
echo "2. Run 'conda activate ml-zoomcamp' to start your environment."
echo "3. Start Jupyter by typing 'jupyter notebook'."
echo "   It will run securely on https://0.0.0.0:8888/"
echo "   (Note: Your browser will warn you about the self-signed certificate. This is normal, and you can safely proceed.)"
echo "=========================================================="