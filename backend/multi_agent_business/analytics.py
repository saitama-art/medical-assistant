import pandas as pd

def analyze_revenue(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=["InvoiceDate"]).copy()
    monthly = (
        valid.assign(Month=valid["InvoiceDate"].dt.to_period("M").astype(str))
        .groupby("Month", as_index=False)
        .agg(NetRevenue=("Revenue", "sum"), LineCount=("InvoiceNo", "size"))
        .sort_values("Month")
    )
    positive = valid[(valid["Quantity"] > 0) & (~valid["IsCancellation"])]
    return {
        "metric": "revenue_and_orders",
        "net_revenue_all_rows": round(float(valid["Revenue"].sum()), 2),
        "positive_order_line_count": int(len(positive)),
        "unique_positive_invoices": int(positive["InvoiceNo"].nunique()),
        "monthly_net_revenue": monthly.to_dict(orient="records"),
        "definition": "净收入按 Quantity*UnitPrice 汇总，退货/取消行可能贡献负值；订单数只统计正数量且非取消发票。",
    }

def analyze_customers(df: pd.DataFrame) -> dict:
    valid = df[
        df["CustomerID"].notna()
        & (df["Quantity"] > 0)
        & (~df["IsCancellation"])
        & (df["UnitPrice"] > 0)
    ].copy()
    customer = valid.groupby("CustomerID").agg(
        revenue=("Revenue", "sum"),
        invoices=("InvoiceNo", "nunique"),
        quantity=("Quantity", "sum"),
    ).sort_values("revenue", ascending=False)
    top = customer.head(10).reset_index()
    return {
        "metric": "customer_contribution",
        "eligible_customer_count": int(customer.shape[0]),
        "top_10_customers": top.round(2).to_dict(orient="records"),
        "top_10_revenue_share_pct": round(
            float(top["revenue"].sum() / customer["revenue"].sum() * 100), 2
        ) if customer["revenue"].sum() else 0,
        "definition": "排除缺失 CustomerID、非正数量、取消发票和非正单价的明细。",
    }

def analyze_products(df: pd.DataFrame) -> dict:
    valid = df[
        (df["Quantity"] > 0)
        & (~df["IsCancellation"])
        & (df["UnitPrice"] > 0)
        & df["Description"].notna()
    ].copy()
    products = valid.groupby("Description").agg(
        quantity=("Quantity", "sum"),
        revenue=("Revenue", "sum"),
        invoices=("InvoiceNo", "nunique"),
    )
    top_revenue = products.sort_values("revenue", ascending=False).head(10)
    top_quantity = products.sort_values("quantity", ascending=False).head(10)
    return {
        "metric": "product_performance",
        "top_10_by_revenue": top_revenue.round(2).reset_index().to_dict(orient="records"),
        "top_10_by_quantity": top_quantity.round(2).reset_index().to_dict(orient="records"),
        "eligible_product_count": int(products.shape[0]),
        "definition": "排除取消/退货行、非正单价和缺少商品描述的明细。",
    }

def analyze_returns_by_month(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=["InvoiceDate"]).copy()
    return_rows = valid[(valid["Quantity"] < 0) | valid["IsCancellation"]].copy()
    return_rows["Month"] = return_rows["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = (
        return_rows.groupby("Month", as_index=False)
        .agg(ReturnOrCancelLines=("InvoiceNo", "size"),
             QuantityMagnitude=("Quantity", lambda s: int(s.abs().sum())),
             ValueMagnitude=("Revenue", lambda s: round(float(s.abs().sum()), 2)))
        .sort_values("Month")
    )
    return {
        "metric": "returns_and_cancellations_by_month",
        "monthly_return_cancel_stats": monthly.to_dict(orient="records"),
        "definition": "退货/取消行数按 Quantity<0 或取消发票标识统计；金额为相关行 Revenue 绝对值之和，是规模指标而非会计净损失。",
    }
