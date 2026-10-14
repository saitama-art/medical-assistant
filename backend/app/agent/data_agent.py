
import os
import json
import operator
from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# =========================
# 1. 初始化
# =========================

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    temperature=0,
)
print(llm,'llm')

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "Online Retail.xlsx"
MAX_VALIDATION_ROUNDS = 2

if not DATA_PATH.exists():
    raise FileNotFoundError(f"找不到真实数据文件：{DATA_PATH}")

df = pd.read_excel(DATA_PATH)

required_columns = {
    "InvoiceNo", "StockCode", "Description",
    "Quantity", "InvoiceDate", "UnitPrice",
    "CustomerID", "Country",
}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"数据缺少必要字段：{missing}")

df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# 保留负数量和取消订单记录，以便分析退货影响
df = df.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])
df = df[df["UnitPrice"] >= 0].copy()


# =========================
# 2. 真实数据分析工具
# =========================

def analyze_revenue():
    """按月计算净销售额、订单数和退货/取消记录影响。"""
    valid = df[df["Quantity"] > 0]
    monthly = (
        df.assign(Month=df["InvoiceDate"].dt.to_period("M").astype(str))
        .groupby("Month")
        .agg(
            net_revenue=("Revenue", "sum"),
            line_count=("InvoiceNo", "size"),
        )
    )

    orders = (
        valid.assign(Month=valid["InvoiceDate"].dt.to_period("M").astype(str))
        .groupby("Month")["InvoiceNo"]
        .nunique()
    )

    monthly["positive_orders"] = orders
    monthly["positive_orders"] = monthly["positive_orders"].fillna(0)
    monthly["net_revenue"] = monthly["net_revenue"].round(2)

    return {
        "monthly": monthly.reset_index().to_dict("records"),
        "cancelled_invoice_lines": int(
            df["InvoiceNo"].str.startswith("C").sum()
        ),
        "negative_quantity_lines": int((df["Quantity"] < 0).sum()),
        "currency": "GBP",
    }


def analyze_customers():
    """分析客户贡献和复购情况。"""
    customer_df = df[
        df["CustomerID"].notna()
        & (df["Quantity"] > 0)
        & (~df["InvoiceNo"].str.startswith("C"))
    ].copy()

    customer_df["CustomerID"] = customer_df["CustomerID"].astype(int)

    customer_stats = customer_df.groupby("CustomerID").agg(
        revenue=("Revenue", "sum"),
        orders=("InvoiceNo", "nunique"),
    )

    top_customers = (
        customer_stats.sort_values("revenue", ascending=False)
        .head(10)
        .round(2)
        .reset_index()
        .to_dict("records")
    )

    return {
        "identified_customers": int(customer_stats.shape[0]),
        "repeat_customers": int((customer_stats["orders"] > 1).sum()),
        "repeat_customer_rate_pct": round(
            (customer_stats["orders"] > 1).mean() * 100, 2
        ),
        "top_customers": top_customers,
        "note": "仅根据存在 CustomerID 的正数量交易计算",
    }


def analyze_products():
    """分析商品和国家的销售表现。"""
    sales = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (~df["InvoiceNo"].str.startswith("C"))
    ].copy()

    top_products = (
        sales.groupby("Description")
        .agg(
            quantity=("Quantity", "sum"),
            revenue=("Revenue", "sum"),
        )
        .sort_values("revenue", ascending=False)
        .head(10)
        .round(2)
        .reset_index()
        .to_dict("records")
    )

    top_countries = (
        sales.groupby("Country")
        .agg(
            revenue=("Revenue", "sum"),
            orders=("InvoiceNo", "nunique"),
        )
        .sort_values("revenue", ascending=False)
        .head(10)
        .round(2)
        .reset_index()
        .to_dict("records")
    )

    return {
        "top_products": top_products,
        "top_countries": top_countries,
    }


def analyze_returns_by_month():
    """验证退货/取消记录是否集中在特定月份。"""
    temp = df.copy()
    temp["Month"] = temp["InvoiceDate"].dt.to_period("M").astype(str)

    monthly = temp.groupby("Month").agg(
        net_revenue=("Revenue", "sum"),
        negative_quantity_lines=(
            "Quantity", lambda x: int((x < 0).sum())
        ),
        total_lines=("InvoiceNo", "size"),
    )

    return monthly.round(2).reset_index().to_dict("records")


# =========================
# 3. Multi-Agent 共享状态
# =========================

class BusinessState(TypedDict):
    question: str
    findings: Annotated[list[dict], operator.add]
    validation_result: str
    validation_reason: str
    validation_round: int
    final_report: str


# =========================
# 4. 三个专业分析 Agent
# =========================

def data_agent(state: BusinessState):
    evidence = analyze_revenue()

    response = llm.invoke(
        "你是经营数据分析师。根据给定的真实计算结果分析月度营收、"
        "订单变化和退货影响。不要编造数字，不要把相关性说成因果关系。\n"
        f"数据：{json.dumps(evidence, ensure_ascii=False, default=str)}"
    )

    return {
        "findings": [{
            "agent": "data_agent",
            "evidence": evidence,
            "analysis": response.content,
        }]
    }


def customer_agent(state: BusinessState):
    evidence = analyze_customers()

    response = llm.invoke(
        "你是客户经营分析师。分析客户贡献、复购率和客户集中度。"
        "说明统计口径和局限，不要从客户贡献直接推断客户流失。\n"
        f"数据：{json.dumps(evidence, ensure_ascii=False, default=str)}"
    )

    return {
        "findings": [{
            "agent": "customer_agent",
            "evidence": evidence,
            "analysis": response.content,
        }]
    }


def product_agent(state: BusinessState):
    evidence = analyze_products()

    response = llm.invoke(
        "你是商品经营分析师。分析畅销商品、国家销售分布和经营机会。"
        "只依据提供的数据，不得声称数据证明了未测量的市场原因。\n"
        f"数据：{json.dumps(evidence, ensure_ascii=False, default=str)}"
    )

    return {
        "findings": [{
            "agent": "product_agent",
            "evidence": evidence,
            "analysis": response.content,
        }]
    }


# =========================
# 5. 验证 Agent
# =========================

def validation_agent(state: BusinessState):
    round_number = state["validation_round"] + 1

    result = llm.invoke(
        "你是独立的经营分析验证员。检查以下发现是否有数据证据支持，"
        "是否存在数字与结论不一致、口径混淆或因果推断过度的问题。\n"
        "如果证据基本充分，第一行输出 SUPPORTED；"
        "如果存在可通过补充交易数据解决的关键缺口，第一行输出 INSUFFICIENT。\n"
        "随后说明理由和需要补查的具体内容。不要把缺乏证据当作证据。\n\n"
        f"研究结果：{json.dumps(state['findings'], ensure_ascii=False, default=str)}"
    ).content

    first_line = result.strip().splitlines()[0].upper()
    status = (
        "supported"
        if first_line.startswith("SUPPORTED")
        else "insufficient"
    )

    return {
        "validation_result": status,
        "validation_reason": result,
        "validation_round": round_number,
    }


# =========================
# 6. 定向补充调查
# =========================

def targeted_research(state: BusinessState):
    # 补充真实交易数据中的退货/取消月度分布
    evidence = analyze_returns_by_month()

    response = llm.invoke(
        "你负责定向补充经营调查。现在已有初步分析，"
        "请结合新增的月度退货/取消数据，核查哪些结论得到加强，"
        "哪些仍然不能证明。不要重新编造或重复生成原始数据。\n"
        f"此前发现：{json.dumps(state['findings'], ensure_ascii=False, default=str)}\n"
        f"补充数据：{json.dumps(evidence, ensure_ascii=False, default=str)}"
    )

    return {
        "findings": [{
            "agent": "targeted_research",
            "evidence": evidence,
            "analysis": response.content,
        }]
    }


def route_after_validation(state: BusinessState):
    if state["validation_result"] == "supported":
        return "report"

    if state["validation_round"] >= MAX_VALIDATION_ROUNDS:
        return "report"

    return "targeted_research"


# =========================
# 7. 生成最终报告
# =========================

def generate_report(state: BusinessState):
    response = llm.invoke(
        "你是经营分析报告负责人。根据所有研究和验证结果生成报告，"
        "包含：核心发现、数据证据、经营建议、分析局限和未解决问题。"
        "明确区分观测事实与推测；不能把相关性当作因果关系。"
        "数据来自2010—2011年的历史电商交易，不代表当前市场状况。\n\n"
        f"研究结果：{json.dumps(state['findings'], ensure_ascii=False, default=str)}\n"
        f"验证状态：{state['validation_result']}\n"
        f"验证说明：{state['validation_reason']}"
    )

    return {"final_report": response.content}


# =========================
# 8. 构建并运行 LangGraph
# =========================

builder = StateGraph(BusinessState)

builder.add_node("data_agent", data_agent)
builder.add_node("customer_agent", customer_agent)
builder.add_node("product_agent", product_agent)
builder.add_node("validation_agent", validation_agent)
builder.add_node("targeted_research", targeted_research)
builder.add_node("generate_report", generate_report)

# 三个独立分析任务并行
builder.add_edge(START, "data_agent")
builder.add_edge(START, "customer_agent")
builder.add_edge(START, "product_agent")

# 汇聚后验证
builder.add_edge("data_agent", "validation_agent")
builder.add_edge("customer_agent", "validation_agent")
builder.add_edge("product_agent", "validation_agent")

# 验证后决定报告或补充调查
builder.add_conditional_edges(
    "validation_agent",
    route_after_validation,
    {
        "report": "generate_report",
        "targeted_research": "targeted_research",
    },
)

builder.add_edge("targeted_research", "validation_agent")
builder.add_edge("generate_report", END)

app = builder.compile()


if __name__ == "__main__":
    result = app.invoke(
        {
            "question": "分析电商业务的营收、客户和商品表现，并提出经营建议。",
            "findings": [],
            "validation_result": "",
            "validation_reason": "",
            "validation_round": 0,
            "final_report": "",
        },
        config={"recursion_limit": 12},
    )

    output_path = Path(__file__).parent / "business_report.md"
    output_path.write_text(result["final_report"], encoding="utf-8")

    print(result["final_report"])
    print(f"\n报告已保存：{output_path}")
    print(f"验证轮次：{result['validation_round']}")
