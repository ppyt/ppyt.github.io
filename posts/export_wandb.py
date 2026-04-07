import os

# ========== 在这里填入你的 W&B API Key ==========
# 从 https://wandb.ai/authorize 获取
# 必须在 import wandb 之前设置环境变量
os.environ["WANDB_API_KEY"] = "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"
# ================================================

import wandb
import pandas as pd

# 导出目录
output_dir = os.path.dirname(os.path.abspath(__file__))

api = wandb.Api()
runs = api.runs("1580259346-tsinghua-university/verl_gsm8k_rl_compare")

# 导出 summary metrics
summary_list = []
for run in runs:
    row = {
        "name": run.name,
        "state": run.state,
        **run.summary._json_dict,
        **{f"config/{k}": v for k, v in run.config.items()}
    }
    summary_list.append(row)

df = pd.DataFrame(summary_list)
summary_path = os.path.join(output_dir, "wandb_summary.csv")
df.to_csv(summary_path, index=False)
print(f"Summary 已导出到: {summary_path}")

# 导出每个 run 的完整训练历史
for run in runs:
    history = run.history()
    # 用 run name 做文件名，替换特殊字符
    safe_name = run.name.replace("/", "_").replace(" ", "_")
    history_path = os.path.join(output_dir, f"wandb_history_{safe_name}.csv")
    history.to_csv(history_path, index=False)
    print(f"History [{run.name}] 已导出到: {history_path}")

print("\n全部导出完成！")
