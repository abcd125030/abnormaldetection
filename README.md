# 1.构建和推送docker镜像
- 在本地构建Docker镜像并推送到Docker Hub。
- ``docker build -t abnormal-detection-service .``
- ``docker push abnormal-detection-service``
# 2.在云服务器上运行容器
- 登录到宋明提供的云服务器，并运行以下命令来启动容器。
- ``docker pull abnormal-detection-service``
- ``docker run -d --name abnormal-detection-service -p 8000:8000 abnormal-detection-service``
# 3.单元测试
- 测试用例数据在data目录，input/是测试用例输入参数，output/是测试用例输出，使用postman完成接口测试