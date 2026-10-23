"""管理员 CLI：创建用户（本期不开放自助注册，决策 #1）。

用法（在 backend/ 目录下）：
    python scripts/create_user.py <username> [--display-name 名字] [--admin]

密码经 getpass 隐式输入并二次确认；哈希为 scrypt（见 app/auth/passwords.py）。
重复用户名给出明确提示并以非零码退出。
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth.passwords import hash_password  # noqa: E402
from app.auth.users import UserStoreError, create_user  # noqa: E402
from app.db import init_db  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="创建医疗助手用户（管理员操作）")
    parser.add_argument("username", help="登录用户名（大小写不敏感，唯一）")
    parser.add_argument("--display-name", default="", help="展示名（可选）")
    parser.add_argument("--admin", action="store_true",
                        help="授予管理员权限（/upload 等维护接口）")
    args = parser.parse_args()

    if not args.username.strip():
        print("用户名不能为空。")
        return 2

    password = getpass.getpass("请输入密码（至少 8 位）: ")
    if len(password) < 8:
        print("密码太短：至少 8 位。")
        return 2
    if password != getpass.getpass("请再次输入密码: "):
        print("两次输入不一致。")
        return 2

    init_db()
    try:
        user_id = create_user(args.username, hash_password(password),
                              display_name=args.display_name,
                              is_admin=args.admin)
    except UserStoreError:
        print(f"创建失败：用户名 '{args.username}' 已存在。")
        return 1
    print(f"用户创建成功：id={user_id} username={args.username}"
          f"{'（管理员）' if args.admin else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
