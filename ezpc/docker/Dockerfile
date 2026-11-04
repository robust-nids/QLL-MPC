# x86 Ubuntu image, compilers, Ocaml, and cpu only pytorch for beacon 
# x86_64 Ubuntu envionment for buliding Beacon/SCI 
# SCI uses x86 and cannot be built on arm64 so we have this on x86 emulator
# mirrors Ubuntu + gcc-9, cmake, libssl, libgmp, Ocaml, menhir/stdint

# run: docker build --platform linux/amd64 -t beacon-sci docker/
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake make curl ca-certificates patch \
        libssl-dev libgmp-dev libmpfr-dev \
        ocaml-nox ocamlbuild ocaml-findlib menhir ocaml-dune \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Build OCaml stdint with Ubuntu
RUN curl -fsSL https://github.com/andrenth/ocaml-stdint/archive/refs/tags/0.7.0.tar.gz | tar -xz -C /tmp \
    && cd /tmp/ocaml-stdint-0.7.0 \
    && dune build -p stdint @install && dune install stdint \
    && echo "$(ocamlfind printconf destdir)/stublibs" >> "$(ocamlfind ocamlc -where)/ld.conf" \
    && rm -rf /tmp/ocaml-stdint-0.7.0

# Beacon frontend (compile_networks.py) usees pytorch 
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
    
RUN pip3 install --no-cache-dir "numpy==1.24.4"

WORKDIR /work
