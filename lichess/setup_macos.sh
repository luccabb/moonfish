#!/bin/bash

set -e  # Exit on error

LICHESS_BOT_DIR="${LICHESS_BOT_DIR:-../lichess-bot}"

# Check if brew is installed
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is not installed. Please install it from https://brew.sh"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "No .env file found. You need to create one with your Lichess API token."
    echo "Please check .env.example for the required format."
    echo "Make sure your token has the 'bot:play' OAuth scope from https://lichess.org/account/oauth/token"
    read -p "Would you like to create .env file now? [Y/n] " create_env
    create_env=$(echo "$create_env" | tr '[:upper:]' '[:lower:]')

    if [[ -z "$create_env" || "$create_env" == "y" || "$create_env" == "yes" ]]; then
        read -p "Enter your Lichess API token (with bot:play scope): " token
        if [ -z "$token" ]; then
            echo "Error: Token cannot be empty."
            exit 1
        fi
        echo "export LICHESS_TOKEN=$token" > .env
        echo ".env file created successfully!"
    else
        echo "Please create .env file manually following .env.example before running this script."
        exit 1
    fi
fi

source .env

# Validate token is set
if [ -z "$LICHESS_TOKEN" ]; then
    echo "Error: LICHESS_TOKEN is not set in .env file."
    exit 1
fi

# Build binary
read -p "Do you want to build a new binary? [Y/n] " answer
answer=$(echo "$answer" | tr '[:upper:]' '[:lower:]')
if [[ -z "$answer" || "$answer" == "y" || "$answer" == "yes" ]]; then
    echo "Building moonfish binary..."
    if ! make build-lichess; then
        echo "Error: Build failed."
        exit 1
    fi
fi

# Verify binary exists
if [ ! -f dist/moonfish ]; then
    echo "Error: dist/moonfish not found. Please build the binary first."
    exit 1
fi

# Install git-lfs if not already installed
if ! command -v git-lfs &> /dev/null; then
    echo "Installing git-lfs..."
    brew install git-lfs
fi
git lfs install
git lfs pull

# Install gettext for envsubst if not already installed
if ! command -v envsubst &> /dev/null; then
    echo "Installing gettext..."
    brew install gettext
fi

# Check if lichess-bot directory exists
if [ ! -d "$LICHESS_BOT_DIR" ]; then
    echo "Error: lichess-bot directory not found at $LICHESS_BOT_DIR"
    echo "Please clone it first:"
    echo "  cd .. && git clone https://github.com/lichess-bot-devs/lichess-bot.git"
    exit 1
fi

# Copy files
echo "Copying files to $LICHESS_BOT_DIR..."
mkdir -p "$LICHESS_BOT_DIR/engines/opening_book"
cp -f dist/moonfish "$LICHESS_BOT_DIR/engines/main"
cp -f opening_book/cerebellum.bin "$LICHESS_BOT_DIR/engines/opening_book/cerebellum.bin"
envsubst < lichess/config.yml > "$LICHESS_BOT_DIR/config.yml"

echo ""
echo "Setup complete! To start playing:"
echo "1. cd $LICHESS_BOT_DIR"
echo "2. python3 lichess-bot.py"
echo ""
