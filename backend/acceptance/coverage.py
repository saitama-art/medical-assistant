"""知识覆盖测试集：P0 饮食营养与体重管理（任务清单第三节 P0，共 20 题）。

目的不是"通过/失败"，而是**定位失败原因**，每轮扩源后复跑对比：
- 检索级（默认，零 LLM）：期望文档是否进 top-k → RETRIEVAL_OK / RETRIEVAL_MISS
  / KNOWLEDGE_MISSING（该主题确认无来源，记「待补充来源」，不伪造覆盖）；
- 链路级（--chat，真实 LLM）：outcome=completed → OK；insufficient_evidence 再按
  检索是否命中细分 EVIDENCE_INSUFFICIENT（检索到了但判定器拒答）与
  RETRIEVAL_MISS（根本没检索到期望文档）；其余 outcome 原样记录。

用法（backend/ 下，ai_agent 环境）：
    python -m acceptance.coverage --group p0     # P0 饮食营养（检索级，零 LLM）
    python -m acceptance.coverage --group p1a    # P1 运动睡眠生活习惯（检索级）
    python -m acceptance.coverage --group p1b    # P1 常见症状与疾病（检索级）
    python -m acceptance.coverage --group p1a --chat   # 追加真实链路（消耗 LLM 额度）
    python -m acceptance.coverage --group p0 --out x.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# ---- P0 饮食营养与体重管理测试集（任务清单第三节 P0 1~20）----
# expect_docs: 判定「已覆盖」所期望命中的文档名子串（空列表 = 已知无来源，记缺口）。
COVERAGE_CASES = [
    dict(id="P0-01", question="我想减肥，饮食如何安排？", topic="体重管理",
         expect_docs=["肥胖"]),
    dict(id="P0-02", question="减肥期间早餐、午餐、晚餐应该怎么吃？", topic="体重管理",
         expect_docs=["肥胖"]),
    dict(id="P0-03", question="减肥期间可以吃米饭、面条和馒头吗？", topic="膳食结构",
         expect_docs=["碳水化合物、蛋白质和脂肪", "肥胖"]),
    dict(id="P0-04", question="减肥期间晚上饿了，可以吃东西吗？", topic="体重管理",
         expect_docs=["肥胖"]),
    dict(id="P0-05", question="减肥期间可以吃水果吗？", topic="膳食结构",
         expect_docs=["碳水化合物、蛋白质和脂肪", "肥胖"]),
    dict(id="P0-06", question="减肥期间体重不下降，可能是什么原因？", topic="体重管理",
         expect_docs=["肥胖"]),
    dict(id="P0-07", question="怎样估算每天需要摄入多少能量？", topic="能量",
         expect_docs=["卡路里"]),
    dict(id="P0-08", question="减肥期间如何保证蛋白质、维生素和矿物质摄入？",
         topic="营养素", expect_docs=["碳水化合物、蛋白质和脂肪", "维生素和矿物质"]),
    dict(id="P0-09", question="普通成年人每天应该怎样搭配饮食？", topic="膳食结构",
         expect_docs=["营养概述", "营养需求"]),
    dict(id="P0-10", question="每天应该吃多少蔬菜和水果？", topic="膳食结构",
         expect_docs=["营养需求", "营养概述"]),
    dict(id="P0-11", question="经常吃外卖，怎样搭配相对健康？", topic="膳食结构",
         expect_docs=["营养概述"]),
    dict(id="P0-12", question="每天喝多少水比较合适？", topic="饮水",
         expect_docs=[], note="待补充来源：MSD 大众版无每日饮水量词条"),
    dict(id="P0-13", question="高蛋白饮食适合所有人吗？", topic="营养素",
         expect_docs=["碳水化合物、蛋白质和脂肪", "膳食补充剂概述"]),
    dict(id="P0-14", question="如何阅读食品营养成分表？", topic="膳食结构",
         expect_docs=[], note="待补充来源：MSD 大众版无营养成分表词条"),
    dict(id="P0-15", question="高血压患者日常饮食应该注意什么？", topic="特殊人群饮食",
         expect_docs=["高血压"]),
    dict(id="P0-16", question="血糖偏高的人应该如何安排饮食？", topic="特殊人群饮食",
         expect_docs=["2型糖尿病"]),
    dict(id="P0-17", question="血脂偏高的人应该少吃哪些食物？", topic="特殊人群饮食",
         expect_docs=["血脂异常", "胆固醇及脂代谢紊乱"]),
    dict(id="P0-18", question="尿酸偏高的人饮食应该注意什么？", topic="特殊人群饮食",
         expect_docs=["痛风（高尿酸血症）"]),
    dict(id="P0-19", question="慢性肾病患者如何根据病情安排饮食？", topic="特殊人群饮食",
         expect_docs=["慢性肾病"]),
    dict(id="P0-20", question="老年人怎样保证蛋白质和营养摄入？", topic="特殊人群饮食",
         expect_docs=["营养需求"]),
]

# ---- P1A 运动、睡眠与生活习惯（任务清单第三节 P1 1~15）----
COVERAGE_P1A = [
    dict(id="P1A-01", question="普通成年人每周运动多少时间比较合适？", topic="运动",
         expect_docs=["锻炼概述"]),
    dict(id="P1A-02", question="初学者如何制定运动计划？", topic="运动",
         expect_docs=["开始锻炼计划"]),
    dict(id="P1A-03", question="减肥应该以有氧运动还是力量训练为主？", topic="运动",
         expect_docs=["选择合适的运动", "锻炼概述"]),
    dict(id="P1A-04", question="久坐人群怎样安排日常活动？", topic="运动",
         expect_docs=["锻炼概述", "锻炼的益处"]),
    dict(id="P1A-05", question="运动后肌肉酸痛正常吗？", topic="运动",
         expect_docs=["肌肉骨骼疼痛"]),
    dict(id="P1A-06", question="运动时出现胸闷、胸痛或头晕应该怎么办？", topic="运动",
         expect_docs=["安全进行锻炼"]),
    dict(id="P1A-07", question="成年人每天需要睡几个小时？", topic="睡眠",
         expect_docs=["睡眠概述"]),
    dict(id="P1A-08", question="入睡困难有哪些常见原因？", topic="睡眠",
         expect_docs=["失眠"]),
    dict(id="P1A-09", question="夜间容易醒来是什么原因？", topic="睡眠",
         expect_docs=["失眠"]),
    dict(id="P1A-10", question="长期熬夜可能产生哪些影响？", topic="睡眠",
         expect_docs=["昼夜节律失调性睡眠障碍", "睡眠概述"]),
    dict(id="P1A-11", question="打鼾严重是否需要就医？", topic="睡眠",
         expect_docs=["打鼾"]),
    dict(id="P1A-12", question="每天喝咖啡多少比较合适？", topic="饮食",
         expect_docs=[], note="待补充来源：MSD 大众版无咖啡因/咖啡量词条"),
    dict(id="P1A-13", question="长时间使用电脑，如何保护眼睛？", topic="用眼",
         expect_docs=[], note="待补充来源：MSD 大众版无眼疲劳/护眼词条"),
    dict(id="P1A-14", question="经常久坐，怎样减少健康风险？", topic="运动",
         expect_docs=["锻炼概述", "锻炼的益处"]),
    dict(id="P1A-15", question="吸烟和饮酒有哪些主要健康风险？", topic="生活方式",
         expect_docs=["吸烟和其他烟草使用", "酒精使用"]),
]

# ---- P1B 常见症状与疾病科普（任务清单第三节 P1 1~22）----
COVERAGE_P1B = [
    dict(id="P1B-01", question="经常头痛是什么原因？", topic="症状",
         expect_docs=["头痛概述"]),
    dict(id="P1B-02", question="反复头晕可能与哪些因素有关？", topic="症状",
         expect_docs=["头晕和眩晕"]),
    dict(id="P1B-03", question="经常疲劳、乏力可能是什么原因？", topic="症状",
         expect_docs=["疲劳"]),
    dict(id="P1B-04", question="咳嗽持续不愈，什么时候需要就医？", topic="症状",
         expect_docs=["成人咳嗽"]),
    dict(id="P1B-05", question="反复腹痛应该注意什么？", topic="症状",
         expect_docs=["慢性腹痛和反复发作性腹痛"]),
    dict(id="P1B-06", question="经常腹胀、消化不良怎么办？", topic="症状",
         expect_docs=["消化不良", "胃肠气胀（腹胀）"]),
    dict(id="P1B-07", question="便秘有哪些常见原因，如何改善？", topic="症状",
         expect_docs=["成人便秘"]),
    dict(id="P1B-08", question="腹泻时如何补充水分？", topic="症状",
         expect_docs=["成人腹泻"]),
    dict(id="P1B-09", question="反复出现皮疹或瘙痒，应该如何处理？", topic="症状",
         expect_docs=["瘙痒", "荨麻疹（风团）"]),
    dict(id="P1B-10", question="心悸可能有哪些原因？", topic="症状",
         expect_docs=["心悸"]),
    dict(id="P1B-11", question="反复低热或发热应该注意什么？", topic="症状",
         expect_docs=["成人发热"]),
    dict(id="P1B-12", question="感冒和流感有什么区别？", topic="疾病",
         expect_docs=["流行性感冒（流感）", "普通感冒"]),
    dict(id="P1B-13", question="病毒感染和细菌感染有什么区别？", topic="疾病",
         expect_docs=["抗生素概述"]),
    dict(id="P1B-14", question="抗生素对病毒感染有效吗？", topic="用药",
         expect_docs=["抗生素概述"]),
    dict(id="P1B-15", question="什么是高血压？有哪些常见危险因素？", topic="疾病",
         expect_docs=["高血压"]),
    dict(id="P1B-16", question="什么是糖尿病？常见类型有哪些？", topic="疾病",
         expect_docs=["糖尿病概述", "2型糖尿病"]),
    dict(id="P1B-17", question="什么是高血脂？血脂异常有什么风险？", topic="疾病",
         expect_docs=["血脂异常", "胆固醇及脂代谢紊乱"]),
    dict(id="P1B-18", question="什么是高尿酸血症和痛风？", topic="疾病",
         expect_docs=["痛风（高尿酸血症）"]),
    dict(id="P1B-19", question="脂肪肝有哪些常见原因？", topic="疾病",
         expect_docs=["脂肪性肝病"]),
    dict(id="P1B-20", question="慢性肾病有哪些常见危险因素？", topic="疾病",
         expect_docs=["慢性肾病"]),
    dict(id="P1B-21", question="骨质疏松有哪些常见风险因素？", topic="疾病",
         expect_docs=["骨质疏松症"]),
    dict(id="P1B-22", question="胃食管反流病有哪些常见表现？", topic="疾病",
         expect_docs=["胃食管反流病"]),
]

CASE_GROUPS = {
    "p0": ("COVERAGE-P0", COVERAGE_CASES),
    "p1a": ("COVERAGE-P1A", COVERAGE_P1A),
    "p1b": ("COVERAGE-P1B", COVERAGE_P1B),
}

# ---- PA P0 就医指引与紧急风险（任务清单第三节 P0 12 题）----
# 注：紧急类问题优先由安全规则拦截（emergency_handled），此处验证兜底证据覆盖。
COVERAGE_PA = [
    dict(id="PA-01", question="出现持续性头痛应该挂什么科？", topic="就医指引",
         expect_docs=["头痛概述"]),
    dict(id="PA-02", question="腹痛、胸闷、咳嗽等症状通常如何选择就诊科室？", topic="就医指引",
         expect_docs=["慢性腹痛和反复发作性腹痛", "胸痛"]),
    dict(id="PA-03", question="看医生前应该准备哪些信息？", topic="就医指引",
         expect_docs=[], note="待补充来源：MSD 大众版无就诊准备词条"),
    dict(id="PA-04", question="如何记录家庭血压和血糖？", topic="就医指引",
         expect_docs=["高血压", "2型糖尿病"]),
    dict(id="PA-05", question="体检发现异常后应该如何安排复查？", topic="就医指引",
         expect_docs=["常见的医疗检查（总览）"]),
    dict(id="PA-06", question="什么时候应该去急诊，而不是继续居家观察？", topic="紧急风险",
         expect_docs=["急救的步骤"]),
    dict(id="PA-07", question="突发胸痛、呼吸困难应该怎么办？", topic="紧急风险",
         expect_docs=["急救的步骤", "胸痛"]),
    dict(id="PA-08", question="突发口角歪斜、肢体无力、言语不清应该怎么办？", topic="紧急风险",
         expect_docs=["脑卒中（中风）"]),
    dict(id="PA-09", question="严重过敏反应有哪些危险表现？", topic="紧急风险",
         expect_docs=["速发严重过敏反应"]),
    dict(id="PA-10", question="意识不清或抽搐时应该怎么办？", topic="紧急风险",
         expect_docs=["癫痫（抽搐）"]),
    dict(id="PA-11", question="大量出血时应该怎样处理？", topic="紧急风险",
         expect_docs=["内出血"]),
    dict(id="PA-12", question="出现严重脱水表现时应该怎么办？", topic="紧急风险",
         expect_docs=["成人腹泻", "成人恶心和呕吐"]),
]

# ---- P1C P1 检查指标与体检报告解释（任务清单第三节 P1 17 题；无新增文档，验证据覆盖）----
COVERAGE_P1C = [
    dict(id="P1C-01", question="白细胞偏高是什么原因？", topic="检查指标",
         expect_docs=["血液疾病的实验室检查（血常规等）"]),
    dict(id="P1C-02", question="血红蛋白偏低意味着什么？", topic="检查指标",
         expect_docs=["缺铁性贫血", "血液疾病的实验室检查（血常规等）"]),
    dict(id="P1C-03", question="血小板偏高需要关注什么？", topic="检查指标",
         expect_docs=["血液疾病的实验室检查（血常规等）"]),
    dict(id="P1C-04", question="空腹血糖偏高怎么办？", topic="检查指标",
         expect_docs=["糖尿病概述", "2型糖尿病"]),
    dict(id="P1C-05", question="糖化血红蛋白反映什么？", topic="检查指标",
         expect_docs=["2型糖尿病", "糖尿病概述"]),
    dict(id="P1C-06", question="总胆固醇、LDL、HDL、甘油三酯分别代表什么？", topic="检查指标",
         expect_docs=["血脂异常", "胆固醇及脂代谢紊乱"]),
    dict(id="P1C-07", question="ALT、AST 偏高可能有哪些原因？", topic="检查指标",
         expect_docs=["肝脏血液检测"]),
    dict(id="P1C-08", question="肌酐偏高意味着什么？", topic="检查指标",
         expect_docs=["肾功能测定"]),
    dict(id="P1C-09", question="eGFR 代表什么？", topic="检查指标",
         expect_docs=["肾功能测定"]),
    dict(id="P1C-10", question="尿蛋白阳性、尿潜血阳性应该怎样理解？", topic="检查指标",
         expect_docs=["肾功能测定", "血尿", "无症状性蛋白尿和血尿"]),
    dict(id="P1C-11", question="尿酸偏高是否一定是痛风？", topic="检查指标",
         expect_docs=["痛风（高尿酸血症）"]),
    dict(id="P1C-12", question="TSH 偏高或偏低可能意味着什么？", topic="检查指标",
         expect_docs=["甲状腺概述（甲功检查）"]),
    dict(id="P1C-13", question="CRP、ESR 升高代表什么？", topic="检查指标",
         expect_docs=["血液疾病的实验室检查（血常规等）"],
         note="无独立炎症指标词条，以血液实验室检查兜底"),
    dict(id="P1C-14", question="BMI 如何计算？腰围和 BMI 有什么区别？", topic="检查指标",
         expect_docs=["肥胖", "代谢综合征"]),
    dict(id="P1C-15", question="家庭血压应该如何正确测量？", topic="检查指标",
         expect_docs=["高血压"]),
    dict(id="P1C-16", question="一次血压偏高能否确诊高血压？", topic="检查指标",
         expect_docs=["高血压"]),
    dict(id="P1C-17", question="多项体检指标同时异常应该怎样看？", topic="检查指标",
         expect_docs=["常见的医疗检查（总览）"]),
]

# ---- P2A 用药安全（任务清单第三节 P2 1~9）----
COVERAGE_P2A = [
    dict(id="P2A-01", question="服药前应该阅读说明书中的哪些信息？", topic="用药安全",
         expect_docs=["药物安全性和有效性概述"]),
    dict(id="P2A-02", question="漏服一次药物应该怎么办？", topic="用药安全",
         expect_docs=["药物安全性和有效性概述", "服药依从性"]),
    dict(id="P2A-03", question="忘记服药后能否加倍补服？", topic="用药安全",
         expect_docs=["用药错误"]),
    dict(id="P2A-04", question="不同药物能否一起服用？", topic="用药安全",
         expect_docs=["药物相互作用"]),
    dict(id="P2A-05", question="服药后出现皮疹、呼吸困难等反应应该怎么办？", topic="用药安全",
         expect_docs=["药物不良反应", "药物安全性和有效性概述"]),
    dict(id="P2A-06", question="抗生素为什么不能随意使用？", topic="用药安全",
         expect_docs=["抗生素概述"]),
    dict(id="P2A-07", question="保健品会不会与处方药发生相互作用？", topic="用药安全",
         expect_docs=["膳食补充剂概述", "药物相互作用"]),
    dict(id="P2A-08", question="长期服药能否自行停药？", topic="用药安全",
         expect_docs=["服药依从性", "药物安全性和有效性概述"]),
    dict(id="P2A-09", question="孕妇、老年人和肝肾功能异常者用药需要注意什么？", topic="用药安全",
         expect_docs=["影响药物反应的因素（年龄/特殊人群）", "孕期用药"]),
]

# ---- P2B 特殊人群（任务清单第三节 P2 1~8）----
COVERAGE_P2B = [
    dict(id="P2B-01", question="儿童发热时，家长应该观察哪些表现？", topic="特殊人群",
         expect_docs=["婴儿和儿童发热"]),
    dict(id="P2B-02", question="老年人如何预防跌倒？", topic="特殊人群",
         expect_docs=["老年人跌倒"]),
    dict(id="P2B-03", question="老年人出现突然意识或行为变化需要注意什么？", topic="特殊人群",
         expect_docs=["谵妄（急性意识模糊）"]),
    dict(id="P2B-04", question="孕期常见不适有哪些一般处理原则？", topic="特殊人群",
         expect_docs=["孕期用药"], note="孕期非药物不适无专门词条，以孕期用药兜底"),
    dict(id="P2B-05", question="产后恢复有哪些基本注意事项？", topic="特殊人群",
         expect_docs=["产后护理概述"]),
    dict(id="P2B-06", question="更年期常见症状有哪些？", topic="特殊人群",
         expect_docs=["更年期"]),
    dict(id="P2B-07", question="慢性病患者如何进行家庭监测？", topic="特殊人群",
         expect_docs=["高血压", "2型糖尿病"]),
    dict(id="P2B-08", question="多种慢性病并存时，为什么需要综合评估？", topic="特殊人群",
         expect_docs=[], note="待补充来源：无多重用药/共病综合评估词条"),
]

CASE_GROUPS.update({
    "pa": ("COVERAGE-PA", COVERAGE_PA),
    "p1c": ("COVERAGE-P1C", COVERAGE_P1C),
    "p2a": ("COVERAGE-P2A", COVERAGE_P2A),
    "p2b": ("COVERAGE-P2B", COVERAGE_P2B),
})

# ---- EXT 阶段 5 补源后的缺口复测（72 题报告 D 类 + 相关题；2026-10-21）----
COVERAGE_EXT = [
    dict(id="EXT-01", question="疫苗接种的基本原则和常见误区有哪些？", topic="预防",
         expect_docs=["免疫接种概述"]),
    dict(id="EXT-02", question="哪些人应该接种流感疫苗？", topic="预防",
         expect_docs=["流感疫苗", "免疫接种概述"]),
    dict(id="EXT-03", question="癌症筛查的基本原则是什么？", topic="预防",
         expect_docs=["癌症（WHO 实况报道）"]),
    dict(id="EXT-04", question="尿蛋白阳性、尿潜血阳性应该怎样理解？", topic="检查指标",
         expect_docs=["血尿", "无症状性蛋白尿和血尿"]),
    dict(id="EXT-05", question="手足口病有哪些表现，什么时候需要就医？", topic="传染病",
         expect_docs=["手足口病"]),
    dict(id="EXT-06", question="多种慢性病并存时，为什么需要综合评估？", topic="慢病",
         expect_docs=["衰老和药物", "老年人的治疗目标"]),
    dict(id="EXT-07", question="孕期常见不适有哪些一般处理原则？", topic="特殊人群",
         expect_docs=["孕期自我保健", "孕期母体变化"]),
]

CASE_GROUPS["ext"] = ("COVERAGE-EXT", COVERAGE_EXT)

# ---- 失败原因分类（纯函数，可单测） ----


def classify(case: dict, hit_rank: int | None, outcome: str | None) -> str:
    """按检索命中与链路 outcome 给出失败/成功分类。

    hit_rank: 期望文档在 top-k 中的名次（1 起），None=未命中。
    outcome: 链路 outcome（None 表示只跑了检索级）。
    """
    if outcome is None:
        # 检索级判定
        if hit_rank is not None:
            return "RETRIEVAL_OK"
        return ("KNOWLEDGE_MISSING" if not case["expect_docs"]
                else "RETRIEVAL_MISS")
    if outcome == "completed":
        return "OK"
    if outcome == "insufficient_evidence":
        return ("EVIDENCE_INSUFFICIENT" if hit_rank is not None
                else ("KNOWLEDGE_MISSING" if not case["expect_docs"] else "RETRIEVAL_MISS"))
    return f"OUTCOME:{outcome}"


# ---- 执行 ----


def _hit_rank(top: list[dict], expect_docs: list[str]) -> int | None:
    for i, chunk in enumerate(top, 1):
        doc = chunk.get("doc") or ""
        if any(k in doc for k in expect_docs):
            return i
    return None


def run_coverage(do_chat: bool, group: str = "p0", client_factory=None) -> dict:
    from .runner import ClientCtx, _safe_record

    if client_factory is None:
        from fastapi.testclient import TestClient
        from app.rag.main import app
        client = ClientCtx(TestClient(app))
    else:
        client = client_factory()

    group_name, cases = CASE_GROUPS[group]
    records = []
    for case in cases:
        rec = dict(case)
        t0 = time.monotonic()
        diag = client.retrieve(case["question"])
        top = diag["top"] or []
        rank = _hit_rank(top, case["expect_docs"])
        rec["hit_rank"] = rank
        rec["top1"] = top[0]["doc"] if top else None
        rec["outcome"] = None
        if do_chat and case["expect_docs"]:
            sid = f"cov-{case['id']}-{datetime.now():%H%M%S}"
            code, body, ms = client.chat(sid, case["question"])
            rec["outcome"] = body.get("outcome")
            rec["http"] = code
            rec["answer"] = (body.get("answer") or "")[:300]
            rec["ms"] = ms
        rec["verdict"] = classify(case, rank, rec["outcome"])
        rec["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
        records.append(_safe_record(rec))
        print(f"{case['id']}  {rec['verdict']:22s} hit={rank} top1={rec['top1']}")
    summary = {}
    for r in records:
        summary[r["verdict"]] = summary.get(r["verdict"], 0) + 1
    return dict(group=group_name, generated_at=datetime.now().isoformat(timespec="seconds"),
                chat=do_chat, summary=summary, records=records)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m acceptance.coverage",
                                 description="知识覆盖测试集（诊断用，不计通过/失败）")
    ap.add_argument("--group", choices=["p0", "p1a", "p1b", "pa", "p1c", "p2a", "p2b", "ext"],
                    default="p0",
                    help="测试集分组（p0 饮食营养 / p1a 运动睡眠 / p1b 症状疾病 / "
                         "pa 就医指引 / p1c 检查指标 / p2a 用药安全 / p2b 特殊人群 / "
                         "ext 补源缺口复测）")
    ap.add_argument("--chat", action="store_true",
                    help="追加真实 LLM 链路判定（默认只跑零 LLM 检索级）")
    ap.add_argument("--out", help="JSON 报告路径（默认 acceptance_results/coverage_<组>_<ts>.json）")
    args = ap.parse_args(argv)

    out = args.out
    if not out:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = Path("acceptance_results")
        d.mkdir(exist_ok=True)
        out = str(d / f"coverage_{args.group}_{ts}.json")

    report = run_coverage(do_chat=args.chat, group=args.group)
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n分类汇总：{report['summary']}")
    print(f"报告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
