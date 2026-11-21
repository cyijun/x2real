# Wall-X（wall-oss-0.5 公开基础权重）适配器修复

这里的文件是在 DGX Spark 上用公开的 `x-square-robot/wall-oss-0.5` 基础权重跑 X2Real 评测时整理出的修复，
**没有接入安装流程**，需要时手工使用。

## 适用条件

- **仅适用于公开基础权重**（Hugging Face `x-square-robot/wall-oss-0.5`，实测 snapshot `b7dc23a8`）。
- 适配器：Policy Space（实测 `6825ed0`）的 `policies/wallx/wall_oss_05_artixon_arm_6a_eef/`。
- Wall-X 推理代码：实测提交 `ce2a1ae`。
- **后训练权重**（在归一化后的 X2Real 数据上微调过的）：把夹爪缩放设回 1
  （`WALLX_GRIPPER_MODEL_SCALE=1`，或 deploy.yaml 里 `gripper_model_scale: 1.0`），并按需关掉首块张开夹爪
  （`WALLX_PREOPEN_CHUNKS=0`）。视觉 MLP 补丁与 Wall-X 代码版本有关，与权重无关。

## 修了什么

| 问题 | 现象 / 原因 | 文件 |
| --- | --- | --- |
| 夹爪单位 | 公开基础权重的夹爪通道是原始单位（归一化器 min≈-0.06、max≈5.5），不是适配器声明的 `normalized_01`（0～1）。送进模型的夹爪状态需乘 5.5，模型输出的夹爪需除以 5.5。 | `wallx_adapter_fix.patch`（`gripper_model_scale: 5.5`） |
| 复位时夹爪闭合 | X2Real 复位时夹爪接近闭合，基础模型只见过以张开夹爪开始的回合，会把闭合夹爪当成“已经抓着东西”。每回合第一块动作原地张开两只夹爪，再交给模型。 | `wallx_adapter_fix.patch`（`preopen_chunks: 1`） |
| 视觉 MLP 权重错位 | Wall-X `ce2a1ae` 加载权重时，`reshape_compatible_state_dict` 先按模型的填充后形状（3420→3456）对融合的 `gate_up_proj` 整体尾部补零，导致 `up` 一半错位 36 行，尾部是随机初始化。补丁让这些张量保持原形状，交给模型自己的补零钩子处理。 | `wallx_patch.py`、`ps_patched.py` |
| 公开 config.yml 不全 | 缺 `data.max_length: 1024` 和 `data.resolution.left_wrist_view: 448`（三路相机都要 448）。 | `ckpt_overlay_config.yml` |

`serve_wallx_fix.sh` 是 Spark 上实际使用的启动脚本原件，里面的路径（`~/workspace/x2real/vla_debug`、
`~/workspace/wall-x`、conda 环境 `wallx`）是那台机器上的，只作参考。

## 用法

以下命令在 x2real 仓库根目录执行，`$FIX` 是你自己选的目录。

1. 复制适配器并打补丁（不要直接改 `3rd/policy_space`：`install.sh` 重跑时要求它是干净检出）：

   ```bash
   FIX=$HOME/wallx_oss05_fix
   cp -r 3rd/policy_space/policies/wallx/wall_oss_05_artixon_arm_6a_eef "$FIX"
   patch -d "$FIX" -p5 < contrib/wallx/wallx_adapter_fix.patch
   ```

   补丁同时把 deploy.yaml 的监听地址改成 `127.0.0.1:8011`，并加上 `gripper_model_scale: 5.5`、`preopen_chunks: 1`。

2. 准备权重目录：把公开权重 snapshot 里的文件软链接到一个新目录，只替换 `config.yml`：

   ```bash
   SNAP=$HOME/.cache/huggingface/hub/models--x-square-robot--wall-oss-0.5/snapshots/<revision>
   CKPT=$HOME/wallx_ckpt_3cam
   mkdir -p "$CKPT" && for f in "$SNAP"/*; do ln -s "$f" "$CKPT/"; done
   rm "$CKPT/config.yml" && cp contrib/wallx/ckpt_overlay_config.yml "$CKPT/config.yml"
   ```

3. 在装好 Wall-X 和 Policy Space 的环境里启动服务。用 `ps_patched.py` 代替 `policy-space` 命令，
   它会先打上视觉 MLP 补丁（`ps_patched.py` 和 `wallx_patch.py` 要放在同一目录）：

   ```bash
   export WALLX_CHECKPOINT_PATH=$CKPT WALLX_ROOT=/path/to/wall-x WALLX_BACKEND=standard
   cd 3rd/policy_space
   python -u ../../contrib/wallx/ps_patched.py serve "$FIX/deploy.yaml" --host 127.0.0.1 --port 8011
   ```

4. 评测端照常运行 `manaenv/scripts/x2real_eval.py`，把 `--server-port` 改成 `8011`。

## 运行时开关

| 环境变量 | 作用 |
| --- | --- |
| `WALLX_GRIPPER_MODEL_SCALE` | 夹爪缩放，覆盖 deploy.yaml（基础权重 5.5；后训练权重 1） |
| `WALLX_PREOPEN_CHUNKS` | 每回合开头原地张开夹爪的块数，覆盖 deploy.yaml（0 关闭） |
| `WALLX_DUMP_PATH` | 设了就把每次推理的状态和动作块追加写入这个 jsonl 文件，用于排查 |
| `WALLX_AGENT_NO_PATCH=1` | 关闭视觉 MLP 补丁 |
