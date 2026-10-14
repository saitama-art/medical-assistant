
def evaluate_findings(
    expected_metrics: list[str],
    findings: list[dict],
) -> dict:
    """检查任务要求的分析结果是否全部返回。"""

    actual_metrics = {
        item.get("metric")
        for item in findings
        if isinstance(item, dict)
    }

    details = {
        metric: {
            "passed": metric in actual_metrics,
            "reason": (
                "已找到对应分析结果"
                if metric in actual_metrics
                else "缺少对应分析结果"
            ),
        }
        for metric in expected_metrics
    }

    total = len(expected_metrics)
    passed = sum(
        result["passed"]
        for result in details.values()
    )

    return {
        "passed": passed == total,
        "score": passed / total if total else 1.0,
        "expected_count": total,
        "passed_count": passed,
        "details": details,
        "actual_metrics": sorted(
            metric
            for metric in actual_metrics
            if isinstance(metric, str)
        ),
    }
