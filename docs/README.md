# 开发约束
1. 每次执行任务、修改前，先把要做的内容拆分成独立 tasks，列在 `docs/NEXT.md` 中，每项 task 使用 `TODO | DOING | DONE | CANCELD` 状态，并在状态变化时及时更新。任务状态变成DONE的时候，从NEXT.md里面移除。任务状态变成CANCELED的时候，在对话中跟开发者确认后删除。
2. 在实现过程中，把已经完成功能写入 `docs/WIKI.md`，作为项目已实现功能索引。
3. 代码功能实现必须与 `spec.md` 保持一致。

# 运行说明

## 环境
- Python `>=3.10`
- 推荐系统：macOS / Linux / Windows

## 安装依赖
```bash
pip install -r requirements.txt
```

## Conda 环境（已验证）
```bash
conda create -y -n vesselme python=3.11
conda run -n vesselme python -m pip install -r requirements.txt
```

## 启动
```bash
python main.py
```
或
```bash
python -m vesselme.main
```

## 核心快捷键
- `B`: 画笔
- `E`: 橡皮
- `Ctrl + 左键拖动`: 调整画笔大小
- `Shift + 左键`: 临时橡皮
- `中键拖动` 或 `Space + 左键`: 平移
- `右键拖动` / 滚轮: 缩放
- `Ctrl+Z / Ctrl+Y`: 撤销 / 重做
- `A`: 显示/隐藏标注层
- `[` / `]`: 缩小 / 增大画笔
- `1~9`: 切换标签
- `S` 或 `Ctrl+S`: 保存当前标签
- `← / →`: 切换上一张/下一张图片

## 锁定与缩放灵敏度
- 右侧标签面板可 `Toggle Lock`：锁定后当前标签禁止绘制、清空、撤销、重做。
- 右侧可调 `Right-drag zoom sensitivity`：数值越小越敏感，越大越平滑。
- 灵敏度设置持久化在 `~/.vesselme_config.json`。
