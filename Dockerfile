FROM debian:stretch

ARG DEBIAN_FRONTEND=noninteractive

#----------------------------------------------------------
# Install common dependencies and create default entrypoint
#----------------------------------------------------------
# replace shell with bash so we can source files
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

ENV LANG="en_US.UTF-8" \
   LC_ALL="C.UTF-8"

RUN apt-get update -qq && apt-get install -yq --no-install-recommends  \
   apt-utils bzip2 ca-certificates curl wget locales unzip tar gcc pigz \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
   && localedef --force --inputfile=en_US --charmap=UTF-8 C.UTF-8 \
   && chmod 777 /opt && chmod a+s /opt

RUN mkdir /root/ROMEO \
   && cd /root/ROMEO/ \
    && wget https://github.com/korbinian90/ROMEO/releases/download/v2.0.2/romeo_linux_2.0.2.tar.gz \
   && tar xzf romeo_linux_2.0.2.tar.gz

ENTRYPOINT ["/root/ROMEO/tmp/romeo_linux_2.0.2/bin/romeo"]
