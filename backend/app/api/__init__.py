"""HTTP 路由层：认证与业务会话接口。

职责边界（设计方案 §3）：本层只做协议适配（请求/响应/Cookie/CSRF），
业务规则在 app/auth/* 与 app/conversations.py。
"""
