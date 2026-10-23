import json
import os

# 精准指向你本地报错的模型配置文件路径
config_path = r"D:\document\py_prj\ImageTextSystem\Smartlmager-sqlite-vec\models\jinaai\jina-embeddings-v4\config.json"

if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 强行给 text_config 注入正确的 mrope 类型，覆盖掉默认的 'default'
    if "text_config" in data:
        data["text_config"]["rope_type"] = "mrope"

    # 上个双保险，外层也注入一个
    data["rope_type"] = "mrope"

    # 写回文件
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print("✅ 模型的 config.json 基因重写成功！这下它绝对认识 mrope 了！")
else:
    print(f"❌ 找不到文件，请检查路径: {config_path}")