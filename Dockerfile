# Use the official CUDA base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /workspace

# Install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    python3 \
    python3-pip \
    clang \
    llvm \
    libomp-dev \
    clangd \
    clang-format

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Install PyTorch and other Python packages
RUN pip3 install torch torchvision torchaudio
RUN pip3 install tqdm matplotlib scikit-learn pandas numpy pyroomacoustics aiohttp aiofiles

# Copy the project files
COPY . /workspace

# Copy the entrypoint script
COPY entrypoint /workspace/entrypoint

# Make the entrypoint script executable
RUN chmod +x /workspace/entrypoint

# Default command
CMD ["bash"]
