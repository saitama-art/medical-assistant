"""密码哈希：hashlib.scrypt（RFC 7914，内存困难型），零第三方依赖。

存储格式自带全部参数，未来调参不影响旧数据校验：
    scrypt$<N>$<r>$<p>$<salt_hex>$<dk_hex>

参数取舍：N=32768, r=8, p=1（约 32MB 内存、单次校验几十毫秒）——
服务端登录校验在此开销可接受，同时显著抬高离线爆破成本。
maxmem 必须显式给足：OpenSSL 默认上限过低会让 N=32768 直接报错。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

_N = 32768
_R = 8
_P = 1
_DKLEN = 64
_MAXMEM = 128 * 1024 * 1024
_SALT_BYTES = 16

_PREFIX = "scrypt"
_FORMAT = "scrypt${N}${r}${p}${salt}${dk}"


def hash_password(password: str) -> str:
    if not isinstance(password, str) or not password:
        raise ValueError("password must be a non-empty str")
    salt = secrets.token_hex(_SALT_BYTES)
    dk = hashlib.scrypt(
        password.encode("utf-8"), salt=bytes.fromhex(salt),
        n=_N, r=_R, p=_P, dklen=_DKLEN, maxmem=_MAXMEM,
    )
    return _FORMAT.format(N=_N, r=_R, p=_P, salt=salt, dk=dk.hex())


def verify_password(password: str, stored: str) -> bool:
    """格式非法或校验失败一律 False，不抛异常（登录路径不允许意外 500）。"""
    try:
        prefix, n_s, r_s, p_s, salt, dk = stored.split("$")
        if prefix != _PREFIX:
            return False
        dk_check = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt),
            n=int(n_s), r=int(r_s), p=int(p_s), dklen=len(bytes.fromhex(dk)),
            maxmem=_MAXMEM,
        )
        return hmac.compare_digest(dk_check.hex(), dk)
    except (ValueError, TypeError, AttributeError):
        return False


__all__ = ["hash_password", "verify_password"]
