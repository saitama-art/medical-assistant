from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(
    model_name=r"F:\models\bge-reranker-base",
    model_kwargs={"device": "cpu"}
)
print("加载成功")