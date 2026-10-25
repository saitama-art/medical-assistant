"""72 道通用健康问题覆盖测试（A01–L06，2026-10-21 冻结）。

阶段 2 执行入口：逐题真实检索（零 LLM）+ 可选真实链路（--chat，消耗 LLM 额度），
记录每题证据（命中文档/距离/chunk 正文摘录/意图/outcome），供 A~F 人工分类：
  A 覆盖充分 / B 覆盖不完整 / C 检索失败 / D 知识缺失 / E 资料可靠性时效 / F 生成引用问题。

用法（backend/ 下，ai_agent 环境）：
    python -m acceptance.health72               # 检索级
    python -m acceptance.health72 --chat        # 检索级 + 真实链路
    python -m acceptance.health72 --out x.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from .promptmeta import prompt_meta  # S4a：报告带 Prompt/词表指纹

# ---- 冻结测试集（12 领域 × 6 题；2026-10-21 经确认冻结，改动需重新确认）----
HEALTH72 = [
    # A 营养、饮食与体重管理
    dict(id="A01", domain="A 营养饮食体重", question="我想减肥，饮食应该如何安排？"),
    dict(id="A02", domain="A 营养饮食体重", question="减肥期间早餐、午餐、晚餐应该怎么吃？"),
    dict(id="A03", domain="A 营养饮食体重", question="怎样估算每天需要摄入多少能量？"),
    dict(id="A04", domain="A 营养饮食体重", question="每天应该吃多少蔬菜和水果？"),
    dict(id="A05", domain="A 营养饮食体重", question="高血压患者日常饮食应该注意什么？"),
    dict(id="A06", domain="A 营养饮食体重", question="老年人怎样保证蛋白质和营养摄入？"),
    # B 运动、睡眠与生活方式
    dict(id="B01", domain="B 运动睡眠", question="普通成年人每周运动多少时间比较合适？"),
    dict(id="B02", domain="B 运动睡眠", question="初学者如何制定运动计划？"),
    dict(id="B03", domain="B 运动睡眠", question="运动时出现胸闷、胸痛或头晕应该怎么办？"),
    dict(id="B04", domain="B 运动睡眠", question="成年人每天需要睡几个小时？"),
    dict(id="B05", domain="B 运动睡眠", question="打鼾严重是否需要就医？"),
    dict(id="B06", domain="B 运动睡眠", question="长期熬夜可能产生哪些影响？"),
    # C 常见慢性病与长期管理
    dict(id="C01", domain="C 慢病管理", question="什么是高血压？有哪些常见危险因素？"),
    dict(id="C02", domain="C 慢病管理", question="什么是糖尿病？常见类型有哪些？"),
    dict(id="C03", domain="C 慢病管理", question="慢性病患者如何进行家庭监测？"),
    dict(id="C04", domain="C 慢病管理", question="长期服药能否自行停药？"),
    dict(id="C05", domain="C 慢病管理", question="什么是高血脂？血脂异常有什么风险？"),
    dict(id="C06", domain="C 慢病管理", question="多种慢性病并存时，为什么需要综合评估？"),
    # D 常见症状与初步就医指导
    dict(id="D01", domain="D 症状就医", question="反复腹痛应该注意什么？"),
    dict(id="D02", domain="D 症状就医", question="经常腹胀、消化不良怎么办？"),
    dict(id="D03", domain="D 症状就医", question="咳嗽持续不愈，什么时候需要就医？"),
    dict(id="D04", domain="D 症状就医", question="反复出现皮疹或瘙痒，应该如何处理？"),
    dict(id="D05", domain="D 症状就医", question="出现持续性头痛应该挂什么科？"),
    dict(id="D06", domain="D 症状就医", question="什么时候应该去急诊，而不是继续居家观察？"),
    # E 体检与实验室检查指标
    dict(id="E01", domain="E 体检指标", question="白细胞偏高是什么原因？"),
    dict(id="E02", domain="E 体检指标", question="血红蛋白偏低意味着什么？"),
    dict(id="E03", domain="E 体检指标", question="总胆固醇、LDL、HDL、甘油三酯分别代表什么？"),
    dict(id="E04", domain="E 体检指标", question="肌酐偏高意味着什么？eGFR 代表什么？"),
    dict(id="E05", domain="E 体检指标", question="尿蛋白阳性、尿潜血阳性应该怎样理解？"),
    dict(id="E06", domain="E 体检指标", question="一次血压偏高能否确诊高血压？"),
    # F 药物基础知识与用药安全
    dict(id="F01", domain="F 用药安全", question="服药前应该阅读说明书中的哪些信息？"),
    dict(id="F02", domain="F 用药安全", question="漏服一次药物应该怎么办？"),
    dict(id="F03", domain="F 用药安全", question="忘记服药后能否加倍补服？"),
    dict(id="F04", domain="F 用药安全", question="不同药物能否一起服用？"),
    dict(id="F05", domain="F 用药安全", question="服药后出现皮疹、呼吸困难等反应应该怎么办？"),
    dict(id="F06", domain="F 用药安全", question="抗生素为什么不能随意使用？"),
    # G 儿童、老年人及特殊人群
    dict(id="G01", domain="G 特殊人群", question="儿童发热时，家长应该观察哪些表现？"),
    dict(id="G02", domain="G 特殊人群", question="老年人如何预防跌倒？"),
    dict(id="G03", domain="G 特殊人群", question="孕期用药需要注意什么？"),
    dict(id="G04", domain="G 特殊人群", question="更年期常见症状有哪些？"),
    dict(id="G05", domain="G 特殊人群", question="孕期常见不适有哪些一般处理原则？"),
    dict(id="G06", domain="G 特殊人群", question="儿童用药剂量需要注意什么？"),
    # H 传染病与感染防护
    dict(id="H01", domain="H 传染病", question="感冒和流感有什么区别？"),
    dict(id="H02", domain="H 传染病", question="病毒感染和细菌感染有什么区别？"),
    dict(id="H03", domain="H 传染病", question="抗生素对病毒感染有效吗？"),
    dict(id="H04", domain="H 传染病", question="哪些人应该接种流感疫苗？"),
    dict(id="H05", domain="H 传染病", question="手足口病有哪些表现，什么时候需要就医？"),
    dict(id="H06", domain="H 传染病", question="病毒性肝炎会通过哪些途径传播？"),
    # I 消化系统与日常常见问题
    dict(id="I01", domain="I 消化常见", question="便秘有哪些常见原因，如何改善？"),
    dict(id="I02", domain="I 消化常见", question="腹泻时如何补充水分和电解质？"),
    dict(id="I03", domain="I 消化常见", question="胃食管反流病有哪些常见表现？"),
    dict(id="I04", domain="I 消化常见", question="幽门螺杆菌感染需要治疗吗？"),
    dict(id="I05", domain="I 消化常见", question="痔疮有哪些表现，如何缓解？"),
    dict(id="I06", domain="I 消化常见", question="脂肪肝有哪些常见原因？"),
    # J 心脑血管健康与急症识别
    dict(id="J01", domain="J 心脑血管", question="突发胸痛、呼吸困难应该怎么办？"),
    dict(id="J02", domain="J 心脑血管", question="如何识别脑卒中（中风）？突发口角歪斜、肢体无力怎么办？"),
    dict(id="J03", domain="J 心脑血管", question="心悸可能有哪些原因？"),
    dict(id="J04", domain="J 心脑血管", question="反复头晕可能与哪些因素有关？"),
    dict(id="J05", domain="J 心脑血管", question="心绞痛和心肌梗死有什么区别？"),
    dict(id="J06", domain="J 心脑血管", question="意识不清或抽搐时应该怎么办？"),
    # K 预防保健与筛查
    dict(id="K01", domain="K 预防筛查", question="普通成年人常规体检应该包括哪些项目？"),
    dict(id="K02", domain="K 预防筛查", question="家庭血压、血糖应该多久测量一次？"),
    dict(id="K03", domain="K 预防筛查", question="哪些人群需要做骨密度检查？"),
    dict(id="K04", domain="K 预防筛查", question="癌症筛查的基本原则是什么？"),
    dict(id="K05", domain="K 预防筛查", question="疫苗接种的基本原则和常见误区有哪些？"),
    dict(id="K06", domain="K 预防筛查", question="普通成年人每天应该怎样搭配饮食才算均衡？"),
    # L 心理健康与其他常见健康问题
    dict(id="L01", domain="L 心理其他", question="抑郁症有哪些常见表现？"),
    dict(id="L02", domain="L 心理其他", question="焦虑到什么程度需要寻求专业帮助？"),
    dict(id="L03", domain="L 心理其他", question="失眠有哪些自助改善方法？"),
    dict(id="L04", domain="L 心理其他", question="经常疲劳、乏力可能是什么原因？"),
    dict(id="L05", domain="L 心理其他", question="戒烟有哪些有效的方法？"),
    dict(id="L06", domain="L 心理其他", question="长期大量饮酒对身体有哪些危害？"),
]


def run(do_chat: bool) -> dict:
    from .runner import ClientCtx, _safe_record
    from fastapi.testclient import TestClient
    from app.rag.main import app

    client = ClientCtx(TestClient(app))
    records = []
    for case in HEALTH72:
        rec = dict(case)
        t0 = time.monotonic()
        diag = client.retrieve(case["question"])
        top = diag["top"] or []
        # 证据摘录：top3 chunk 正文截断；完整 top10 文档名+距离
        rec["top_docs"] = [{"doc": c["doc"], "distance": round(c["distance"], 4)
                            if c.get("distance") is not None else None}
                           for c in top]
        rec["evidence"] = [
            {"rank": i + 1, "doc": c["doc"],
             "distance": round(c["distance"], 4) if c.get("distance") is not None else None,
             "excerpt": (c.get("content") or "")[:300]}
            for i, c in enumerate(top[:3])
        ]
        rec["category"] = None
        rec["outcome"] = None
        if do_chat:
            code, body, ms = client.chat(f"h72-{case['id']}-{datetime.now():%H%M%S}",
                                         case["question"])
            rec["http"] = code
            rec["category"] = body.get("category")
            rec["outcome"] = body.get("outcome")
            rec["risk_notice"] = bool(body.get("risk_notice"))
            rec["answer"] = (body.get("answer") or "")[:400]
            rec["n_sources"] = len(body.get("sources") or [])
            rec["ms"] = ms
        rec["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
        records.append(_safe_record(rec))
        print(case["id"], rec.get("outcome") or "-",
              rec["top_docs"][0]["doc"] if rec["top_docs"] else "EMPTY")
    return dict(group="HEALTH72", frozen="2026-10-21", generated_at=datetime.now().isoformat(timespec="seconds"),
                chat=do_chat, n=len(records), records=records,
                promptmeta=prompt_meta())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m acceptance.health72",
                                 description="72 道通用健康问题覆盖测试（阶段 2 证据采集）")
    ap.add_argument("--chat", action="store_true", help="追加真实链路（消耗 LLM 额度）")
    ap.add_argument("--out", help="JSON 报告路径（默认 acceptance_results/health72_<ts>.json）")
    args = ap.parse_args(argv)

    out = args.out
    if not out:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = Path("acceptance_results")
        d.mkdir(exist_ok=True)
        out = str(d / f"health72_{ts}.json")

    report = run(do_chat=args.chat)
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成 {report['n']} 题，报告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
