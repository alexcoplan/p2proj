# This Dockerfile provides a build environment which should run anywhere.
# 
# One reason this is necessary is because Valgrind is not supported on MacOS
# Sierra (my development environment of choice).

FROM ubuntu:16.04
MAINTAINER Alex Coplan <lexy0202@gmail.com>

CMD ["/bin/bash"]

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update
RUN apt-get -y install build-essential valgrind scons

RUN mkdir -p /root/build
VOLUME /root/build

# clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
