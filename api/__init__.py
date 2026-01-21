# -*- coding: utf-8 -*-
# @Time : 2024/11/12 14:41
# @File : __init__.py
# @Author : lisongming
from sanic import Blueprint
from api.views.by import views_bp

api = Blueprint.group(views_bp)
