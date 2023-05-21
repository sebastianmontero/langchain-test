#!/usr/bin/env bash

usage="./start-db [build|image]"
if [ $# -ne 1 ]; then
    echo $usage
    exit 1
fi

if [[ $1 != 'build' && $1 != 'image' ]]; then
    echo $usage
    exit 1
fi

script_path="$(dirname $(realpath ${BASH_SOURCE[0]}))"

pushd $script_path

# cp .env.all .env

if [[ $1 = build ]]; then
  docker-compose -f docker-compose.db.yaml -f docker-compose.db.build.yaml -p governance-db-$1 up --build -d
else
  docker-compose -f docker-compose.db.yaml -f docker-compose.db.image.yaml -p governance-db-$1 up -d
fi

popd
