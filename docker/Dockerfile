# Ubuntu version: bionic, focal

ARG CMAKE_VERSION="3.20.0"

ARG VARIANT=buster
FROM debian:${VARIANT} AS cmake-install
ARG CMAKE_REPO="https://github.com/Kitware/CMake"
ARG CMAKE_TARGET="Linux-x86_64"
ARG CMAKE_VERSION

RUN mkdir -p /opt/tools/cmake  \

# Install modern CMake
RUN cd /tmp \
    && apt update \
    && apt install -y wget \
    && wget -O cmake.sh ${CMAKE_REPO}/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-${CMAKE_TARGET}.sh \
    && sh ./cmake.sh --prefix=/opt/tools/cmake/ --skip-license \
    && rm ./cmake.sh

FROM tensorflow/tensorflow:2.3.0-custom-op-ubuntu16

#RUN apt-get remove -y cmake

RUN mkdir -p "/opt/tools/"

COPY --from=cmake-install /opt/tools/cmake/ /opt/tools/cmake/
ENV PATH="/opt/tools/cmake/bin:${PATH}"

RUN rm /usr/local/bin/python3.7
