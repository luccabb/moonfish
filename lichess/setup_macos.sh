#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
    echo "No .env file found. You need to create one with your Lichess API token."
    echo "Please check .env.example for the required format."
    echo "Make sure your token has the 'bot:play' OAuth scope from https://lichess.org/account/oauth/token"
    read -p "Would you like to create .env file now? (y/n) " create_env
    create_env=$(echo "$create_env" | tr '[:upper:]' '[:lower:]')
    
    if [[ "$create_env" == "y" || "$create_env" == "yes" ]]; then
        read -p "Enter your Lichess API token (with bot:play scope): " token
        echo "export LICHESS_TOKEN=$token" > .env
        echo ".env file created successfully!"
    else
        echo "Please create .env file manually following .env.example before running this script."
        exit 1
    fi
fi

source .env

# build binary
read -p "Do you want to build a new binary? (y/n) " answer
answer=$(echo "$answer" | tr '[:upper:]' '[:lower:]')
if [[ "$answer" == "y" || "$answer" == "yes" ]]; then
    make build-lichess
fi

# fetch opening book if not already downloaded
brew install git-lfs
git lfs install
git lfs pull

# install gettext for envsubst
brew install gettext

# copy files
if [ -f ../lichess-bot/engines/main ]; then
    rm ../lichess-bot/engines/main
fi
cp dist/moonfish ../lichess-bot/engines/main
mkdir -p ../lichess-bot/engines/opening_book
cp opening_book/cerebellum.bin ../lichess-bot/engines/opening_book/cerebellum.bin
bash .env
envsubst < lichess/config.yml > ../lichess-bot/config.yml

echo ""
echo "Setup complete! To start playing:"
echo "1. cd ../lichess-bot"
echo "2. python3 lichess-bot.py"
echo ""
