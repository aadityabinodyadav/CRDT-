FROM ubuntu:22.04

# Avoid timezone prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and ns-3 dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    python3 \
    python3-dev \
    python3-pip \
    cmake \
    ninja-build \
    git \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Download and compile ns-3.37
WORKDIR /usr/local/src

# ns-3.37 release
RUN wget https://www.nsnam.org/releases/ns-allinone-3.37.tar.bz2 && \
    tar xjf ns-allinone-3.37.tar.bz2 && \
    rm ns-allinone-3.37.tar.bz2

WORKDIR /usr/local/src/ns-allinone-3.37/ns-3.37

# Copy our project files
COPY scratch/crdt_mesh.cc scratch/crdt_mesh.cc

# Configure and build ns-3 optimized version
RUN ./ns3 configure --build-profile=optimized --disable-werror
RUN ./ns3 build

# Default command when container runs: run our mesh app
CMD ["./ns3", "run", "scratch/crdt_mesh"]
