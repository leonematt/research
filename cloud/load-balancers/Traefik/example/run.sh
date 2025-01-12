#!/bin/bash

main() {

    docker network create proxy

    docker compose down --rmi all

    rm config/acme.json

    touch config/acme.json

    chmod 600 config/acme.json

    cd site

    docker build --force-rm -t site .

    cd ..

    docker compose up

}

main "$@"
