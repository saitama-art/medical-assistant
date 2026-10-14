
EVALUATION_DATASET = [
    {
        "case_id": "revenue_001",
        "question": "分析整体净收入、正向订单行数和月度收入趋势。",
        "expected_metrics": [
            "revenue_and_orders",
        ],
        "checks": [
            "净收入与参考计算结果一致",
            "正向订单行数符合统计口径",
            "月度收入趋势有数据支持",
        ],
    },
    {
        "case_id": "customer_001",
        "question": "分析贡献最高的客户，并说明其收入贡献。",
        "expected_metrics": [
            "customer_contribution",
        ],
        "checks": [
            "客户排名与参考计算结果一致",
            "收入贡献比例正确",
        ],
    },
    {
        "case_id": "product_001",
        "question": "找出收入最高的商品，并分析商品销售表现。",
        "expected_metrics": [
            "product_performance",
        ],
        "checks": [
            "商品排名与参考计算结果一致",
            "收入和数量指标正确",
        ],
    },
    {
        "case_id": "business_001",
        "question": "综合分析营收、客户贡献和商品表现。",
        "expected_metrics": [
            "revenue_and_orders",
            "customer_contribution",
            "product_performance",
        ],
        "checks": [
            "相关分析结果均已获取",
            "综合结论与分析数据一致",
        ],
    },
]
