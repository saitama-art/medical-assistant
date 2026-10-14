from evaluation.dataset import EVALUATION_DATASET
from evaluation.evaluator import evaluate_findings

# graph 是你现有的、已经编译好的 LangGraph 图
case = EVALUATION_DATASET[0]

result = graph.invoke({
    "question": case["question"],
    "findings": [],
})

evaluation = evaluate_findings(
    expected_metrics=case["expected_metrics"],
    findings=result.get("findings", []),
)

print(evaluation)