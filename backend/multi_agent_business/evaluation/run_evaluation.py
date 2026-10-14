from graph import build_graph
from evaluation.dataset import EVALUATION_DATASET
from evaluation.evaluator import evaluate_findings


def run_evaluation():
    # 复用现有的图构建逻辑
    graph = build_graph()

    for case in EVALUATION_DATASET:
        print(f"\n{'=' * 50}")
        print(f"评估用例：{case['case_id']}")
        print(f"测试问题：{case['question']}")

        result = graph.invoke({
            "question": case["question"],
            "findings": [],
        })

        evaluation = evaluate_findings(
            expected_metrics=case["expected_metrics"],
            findings=result.get("findings", []),
        )

        print(f"评估结果：{evaluation}")


if __name__ == "__main__":
    run_evaluation()