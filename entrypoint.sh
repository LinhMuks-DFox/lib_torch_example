#!/bin/bash

# Check if libtorch exists
if [ ! -d "/workspace/libtorch" ]; then
  echo "libtorch not found, downloading..."
  curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu121.zip
  unzip libtorch.zip -d /workspace
  rm libtorch.zip
else
  echo "libtorch found, skipping download."
fi

# Check if fmt exists
if [ ! -d "/workspace/fmt" ]; then
  echo "fmt not found, cloning..."
  git clone https://github.com/fmtlib/fmt.git /workspace/fmt
else
  echo "fmt found, skipping clone."
fi

# Run the default command
exec "$@"
