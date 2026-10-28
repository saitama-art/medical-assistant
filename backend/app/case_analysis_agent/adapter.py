"""``app.case_analysis.service`` 的薄适配层。

存在的唯一理由：把领域异常翻译成 Agent 流程可分支处理的机器状态码，
并集中记录"本 Agent 实际依赖的 service 接口清单"。本模块：

- 不含业务逻辑、不伪造实现、不绕过 repository 的 SQL 归属校验；
- 异常一律映射为机器可读短码，不把异常消息（可能含输入值）写进 trace；
- 每个函数返回 ``(value, error_code)``：error_code 为空串表示成功；
  列表类接口出错时 value 为 None，空结果仍是 ``([], "")``，两者不得混用。

接口对应关系（2026-10-28 对照 case_analysis/service.py 实读，逐函数核实）：

    get_patient                  -> service.get_patient
    save_analysis_run            -> service.save_analysis_run
    update_analysis_run          -> service.update_analysis_run
    save_candidate_associations  -> service.save_candidate_associations
    get_patient_history          -> service.get_patient_history
    get_patient_timeline         -> service.get_patient_timeline
    get_patient_observations     -> service.get_patient_observations
    get_patient_medications      -> service.get_patient_medications
    list_encounters              -> service.list_encounters
    compare_observations         -> service.compare_observations
    find_temporal_associations   -> service.find_temporal_associations
"""

from __future__ import annotations

from ..case_analysis import service
from ..case_analysis.models import (
    CaseAccessDeniedError,
    CaseAnalysisError,
    CaseValidationError,
)


def _call(fn, *args, **kwargs):
    """统一异常翻译。access_denied 与 not_found 合并为一个码：本层遵守
    service 的防枚举约定，不向 Agent 流程暴露二者差异。"""
    try:
        return fn(*args, **kwargs), ""
    except CaseAccessDeniedError:
        return None, "access_denied"
    except CaseValidationError:
        return None, "validation_failed"
    except CaseAnalysisError:
        return None, "case_analysis_error"
    except Exception:
        # sqlite3.Error 等存储层故障；异常原文不进 trace。
        return None, "storage_error"


# ==================== 权限与运行登记 ====================


def get_patient(patient_id: str, owner_user_id: int):
    return _call(service.get_patient, patient_id, owner_user_id)


def save_analysis_run(patient_id: str, owner_user_id: int, *,
                      time_range_start=None, time_range_end=None,
                      input_summary: str = ""):
    return _call(
        service.save_analysis_run, patient_id, owner_user_id,
        time_range_start=time_range_start, time_range_end=time_range_end,
        input_summary=input_summary,
    )


def update_analysis_run(run_id: str, owner_user_id: int, *,
                        status: str, error_code: str = ""):
    return _call(
        service.update_analysis_run, run_id, owner_user_id,
        status=status, error_code=error_code,
    )


def save_candidate_associations(run_id: str, owner_user_id: int, drafts):
    return _call(service.save_candidate_associations, run_id, owner_user_id, drafts)


# ==================== 历史数据读取 ====================


def get_patient_history(patient_id: str, owner_user_id: int, *,
                        fact_types=None, start=None, end=None,
                        review_status=None, limit: int = 1000):
    return _call(
        service.get_patient_history, patient_id, owner_user_id,
        fact_types=fact_types, start=start, end=end,
        review_status=review_status, limit=limit,
    )


def get_patient_timeline(patient_id: str, owner_user_id: int):
    return _call(service.get_patient_timeline, patient_id, owner_user_id)


def get_patient_observations(patient_id: str, owner_user_id: int, *,
                             indicator=None, start=None, end=None,
                             limit: int = 1000):
    return _call(
        service.get_patient_observations, patient_id, owner_user_id,
        indicator=indicator, start=start, end=end, limit=limit,
    )


def get_patient_medications(patient_id: str, owner_user_id: int, *,
                            start=None, end=None, limit: int = 1000):
    return _call(
        service.get_patient_medications, patient_id, owner_user_id,
        start=start, end=end, limit=limit,
    )


def list_encounters(patient_id: str, owner_user_id: int, *,
                    start=None, end=None, limit: int = 500):
    return _call(
        service.list_encounters, patient_id, owner_user_id,
        start=start, end=end, limit=limit,
    )


# ==================== 确定性分析 ====================


def compare_observations(patient_id: str, owner_user_id: int, *,
                         indicator=None, start=None, end=None):
    return _call(
        service.compare_observations, patient_id, owner_user_id,
        indicator=indicator, start=start, end=end,
    )


def find_temporal_associations(patient_id: str, owner_user_id: int, *,
                               start=None, end=None):
    return _call(
        service.find_temporal_associations, patient_id, owner_user_id,
        start=start, end=end,
    )
