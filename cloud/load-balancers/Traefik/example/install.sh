#!/bin/bash

command_exists() {

    if command -v $1 >/dev/null 2>&1; then

        echo "$1 is already installed"

        return 0

    else

        echo "$1 needs to be installed"

        return 1

    fi

}

install_dependencies() {

    sudo apt-get update

    sudo apt-get install -y ca-certificates curl

    sudo install -m 0755 -d /etc/apt/keyrings

    if ! command_exists docker; then

        sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc

        sudo chmod a+r /etc/apt/keyrings/docker.asc

        echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

        sudo apt-get update

        sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    fi

    if ! command_exists node; then

        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        
        sudo apt-get install -y nodejs

    fi

    sudo apt-get update

}

main() {
    
    echo "Starting Installation"

    install_dependencies

    echo "Finished installing dependencies"

}

main "$@"
