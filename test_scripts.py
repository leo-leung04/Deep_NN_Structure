from huggingface_hub import list_models

# 尝试获取所有模型（不加过滤条件），然后打印前 10 个模型的基本信息
all_models = list(list_models(limit=50))
for model in all_models[:10]:
    print(f"Model ID: {model.modelId}, Library: {model.library_name}, Task: {model.pipeline_tag or model.task}")
    