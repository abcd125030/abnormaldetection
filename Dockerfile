#!/bin/bash
#
# 基于镜像基础
FROM ubuntu:20.04

# 先安装时间
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV LANGUAGE C.UTF-8

ENV TimeZone=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TimeZone /etc/localtime && echo $TimeZone > /etc/timezone

# 更新包列表并安装编译所需的工具
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libreadline-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libxml2-dev \
    libxslt1-dev \
    curl \
    telnet \
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量以避免设置交互式终端
ENV DEBIAN_FRONTEND=noninteractive

# 确保安装 gcc
RUN apt-get update && apt-get install -y gcc

# 下载 Python 3.11 源代码
# RUN curl -O https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz
RUN curl -O https://repo.huaweicloud.com/python/3.11.0/Python-3.11.0.tgz

# 解压源代码包
RUN tar xzf Python-3.11.0.tgz

# 进入 Python 源代码目录并进行配置、编译和安装
RUN cd Python-3.11.0 && \
    ./configure --enable-optimizations && \
    make -j 8 && \
    make altinstall

# 清理不再需要的文件
RUN rm -rf Python-3.11.0.tgz Python-3.11.0

# 设置Python 3.11作为默认Python版本
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3.11 python3.11 /usr/local/bin/python3.11 1

# 设置pip的Python版本指向
RUN rm /usr/bin/python
RUN ln -s /usr/local/bin/python3.11 /usr/bin/python
RUN ln -s /usr/local/bin/pip3.11 /usr/bin/pip3
RUN ln -s /usr/local/bin/pip3.11 /usr/bin/pip

# 验证Python和pip版本
RUN python --version
RUN pip --version

# 设置代码文件夹工作目录 /abnormaldetection
WORKDIR /abnormaldetection
# 复制当前代码文件到容器中 /abnormaldetection
ADD . /abnormaldetection

# 安装所需的包
RUN pip install -i  https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 使用8080 端口
EXPOSE 8080
# Run app.py when the container launches
CMD ["python", "start.py"]