# SimulacraBench 提交步骤（需要你本人操作）

所有 zip 都在 `~/workSpace/simulacra-bench/zips/`，都已通过 `check_submission_zip.py`，并且在 sandbox 上用官方 `score.py` 跑过 3 个问卷都是 PASS。

## A. 注册（只做一次）
1. 打开 https://www.codabench.org/competitions/17822/ ，用你自己的账号登录/注册。
2. 点 **Terms**，阅读并同意条款；点 **Participate / Register**，等待批准（如需）。
3. 记下你的 **Team ID / 用户名**（排行榜上显示的名字），写报告时要用。

## B. 开发期提交（截止：北京时间 11-14 周六 20:00 = AoE 11-13 结束；每天 3 次，失败不计数）
按顺序提交，每个提交完等结果出来再交下一个：
1. `zips/v1_backoff.zip` —— 官方计数回退基线（无 LLM），作为对照。
2. `zips/v3_ens.zip` —— 主模型（多任务神经网络集成 + 计数回退 + CV stacking + 门控规则）。
3. （第二天）根据结果再交改进版（我会在 STATUS 文件里写明是哪个 zip）。

操作：Codabench 比赛页 → **My Submissions** → 选择 **Development** 阶段 → **Upload** 选 zip → 等待状态变成 Finished → 在 **Results** 看分数。
注意：开发期分数是加噪声 + 四舍五入到 0.01 的，不要只看 0.01 以内的差别。

把每次的分数（总分 + 3 个问卷分数）发给我，或者写进 `~/workSpace/STATUS-neurips.md`。

## C. 决赛提交（北京时间 11-15 周日 20:00 开始 ~ 11-20 周五 20:00 截止；只能成功提交 1 次）
1. 在 Codabench 选 **Final** 阶段，上传 `zips/final.zip`（我会准备好；默认 = 开发期最好的版本，时间预算放大到 3600 秒的版本）。
2. 只能成功 1 次，**不要**交错文件。

## D. OpenReview 技术报告（截止：北京时间 11-20 周五 18:59 = 2026-11-20 10:59 UTC）
1. 打开 https://openreview.net/group?id=NeurIPS.cc/2026/Competition/SimulacraBench ，登录（所有作者都需要 OpenReview 账号）。
2. 代码要公开：在你的 GitHub 建一个公开仓库（不要在仓库名里写 "kaggle"），把 `~/workSpace/simulacra-bench/`（不含 `.venv`、`env/`、`deploy/`、`starter-kit/_sandbox*`）推上去，MIT 协议。
3. 编译 `writeup/report.tex`（Overleaf 上传 `writeup/` 整个目录即可；需要 `neurips_2026.sty`，已放在里面），把决赛分数填进 Table 3（报告要求和 Codabench 决赛分数一致），把 Team ID 填进 `\teamid`。
4. 填表：title、authors、**team_id（和 Codabench 排行榜一致）**、keywords、abstract、PDF、code_url（GitHub 链接）、code_license=MIT、base_models=`None`、external_data=`None`、勾选数据使用声明和两个授权选项 → Submit。
