"""认证子包：密码哈希、服务端登录会话、FastAPI 依赖。

边界：本包只负责"你是谁"（认证）与登录态管理；业务会话的归属校验在
conversations.py（授权）。token 的签发/校验/吊销在 sessions.py，
数据库只存 token 的 SHA-256 摘要。
"""
