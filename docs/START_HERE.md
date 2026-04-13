# START HERE

这是项目的最短启动路径。

## 1) 进入项目目录
```bash
cd .../vesselme
```

## 2) 使用 conda 环境（推荐）
如果你还没有环境：
```bash
conda create -y -n vesselme python=3.11
conda run -n vesselme python -m pip install -r requirements.txt
```

如果环境已创建：
```bash
conda activate vesselme
```

## 3) 启动应用
```bash
python main.py
```

或：
```bash
python -m vesselme.main
```

## 4) 启动后第一步
1. 点击 `File -> Open Folder` 选择眼底图像目录。
2. 左侧选中图片，右侧选中/创建标签后开始标注。

## 5) 常用快捷键
- `B`：画笔
- `E`：橡皮
- `Ctrl + 鼠标滚轮`：调整画笔大小
- `鼠标右键`：临时橡皮
- `中键拖动` 或 `Space + 左键`：平移
- `滚轮`：缩放
- `A`：显示/隐藏标注层
- `Ctrl+Z / Ctrl+Y`：撤销 / 重做
- `[` / `]`：缩小 / 增大画笔
- `1~9`：切换标签
- `S` 或 `Ctrl+S`：保存当前标签
- `← / →`：切换上一张 / 下一张图片

## 6) 锁定
- 右侧 `Toggle Lock` 可锁定当前标签，锁定后禁止绘制、清空、撤销、重做。
