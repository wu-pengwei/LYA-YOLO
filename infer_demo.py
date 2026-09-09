"""
infer_demo.py
LYA‑YOLO 推理演示脚本
仅用于模型推理效果对比，无训练相关代码
权重文件：LYA‑YOLO‑S.pt 、yolov8s.pt
运行前请确认 weights 目录下存在对应pt权重
demo_images目录存放待测试图片
输出结果保存到 run_infer/ours 与 run_infer/baseline
"""

import os
from ultralytics import YOLO

# ====================== 配置区，根据自己本地路径修改 ======================
WEIGHT_OURS = "./weights/LYA‑YOLO‑S.pt"       # 自己模型权重
WEIGHT_BASELINE = "./weights/yolov8s.pt"    # 基线YOLOv8‑S权重
SOURCE_DIR = "./demo_images/"               # 待推理图片文件夹
OUTPUT_ROOT = "./run_infer"                 # 推理输出根目录
CONF_THRESH = 0.25                          # 置信度阈值
IOU_THRESH = 0.45                           # NMS iou阈值
# ========================================================================

def check_file_exist(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在：{filepath}\n请检查权重路径是否正确！")


def main():
    # 校验权重文件
    check_file_exist(WEIGHT_OURS)
    check_file_exist(WEIGHT_BASELINE)
    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"待推理图片目录不存在：{SOURCE_DIR}，请创建并放入测试图片")

    out_ours = os.path.join(OUTPUT_ROOT, "ours")
    out_baseline = os.path.join(OUTPUT_ROOT, "baseline")
    os.makedirs(out_ours, exist_ok=True)
    os.makedirs(out_baseline, exist_ok=True)

    print("==== 加载 LYA‑YOLO‑S 模型 ====")
    model_ours = YOLO(WEIGHT_OURS)
    # 推理：自己模型
    model_ours.predict(
        source=SOURCE_DIR,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        save=True,
        save_txt=False,
        save_conf=True,
        project=out_ours,
        name="",
        exist_ok=True
    )

    print("\n==== 加载 YOLOv8‑S 基线模型 ====")
    model_baseline = YOLO(WEIGHT_BASELINE)
    # 推理：基线模型
    model_baseline.predict(
        source=SOURCE_DIR,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        save=True,
        save_txt=False,
        save_conf=True,
        project=out_baseline,
        name="",
        exist_ok=True
    )

    print(f"\n✅推理完成！")
    print(f"LYA‑YOLO‑S结果输出目录：{out_ours}")
    print(f"YOLOv8‑S基线结果输出目录：{out_baseline}")


if __name__ == "__main__":
    main()
