FROM python:3.6
ENV LC_ALL=C.UTF-8

RUN apt-get update -y && \
    apt-get install -yq wget build-essential gcc zlib1g-dev ca-certificates && \
    apt-get install -y apt-utils make curl fonts-ipafont wamerican gnupg2 && \
    apt-get install -yq git xz-utils file swig sudo software-properties-common

RUN apt-get update \
    && apt-get install -y locales \
    && locale-gen ja_JP.UTF-8 \
    && echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc

RUN python3.6 -m pip install pip --upgrade
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

CMD ["/bin/bash"]
