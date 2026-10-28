"""纵向病例分析基础设施：数据模型、持久化、确定性分析与对外服务接口。

另一终端（病例分析 Agent）只应依赖两个入口：

    from app.case_analysis import service      # 患者/事实/分析的服务函数
    from app.case_analysis import models       # 不可变数据结构与枚举

不得直接访问 repository 的 SQL 层或 db.connect()。
"""

from . import models, service  # noqa: F401
