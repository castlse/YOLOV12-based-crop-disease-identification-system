import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ttkbootstrap import Style
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class YOLOv12App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv12 农作物病害检测系统")
        self.root.geometry("1000x700")

        self.model_path = tk.StringVar()
        self.source_path = tk.StringVar()
        self.yaml_path = tk.StringVar()
        self.save_dir = tk.StringVar()

        self.style = Style("flatly")
        self.create_widgets()



    def create_widgets(self):
        # 上方控件
        frame_top = ttk.Frame(self.root)
        frame_top.pack(fill="x", padx=10, pady=10)

        # 模型权重路径
        ttk.Label(frame_top, text="模型权重路径:").grid(row=0, column=0, sticky=W)
        ttk.Entry(frame_top, textvariable=self.model_path, width=50).grid(
            row=0, column=1, padx=5, sticky=EW)
        ttk.Button(frame_top, text="选择模型", command=self.browse_model).grid(
            row=0, column=2, padx=5)

        # 检测文件
        ttk.Label(frame_top, text="检测文件:").grid(row=1, column=0, sticky=W, pady=5)
        ttk.Entry(frame_top, textvariable=self.source_path, width=50).grid(
            row=1, column=1, padx=5, sticky=EW)
        ttk.Button(frame_top, text="选择图片/视频", command=self.browse_source).grid(
            row=1, column=2, padx=5)

        # 结果保存路径
        ttk.Label(frame_top, text="结果保存路径:").grid(row=2, column=0, sticky=W)
        ttk.Entry(frame_top, textvariable=self.save_dir, width=50).grid(
            row=2, column=1, padx=5, sticky=EW)
        ttk.Button(frame_top, text="选择保存目录", command=self.browse_save_dir).grid(
            row=2, column=2, padx=5)

        # 开始检测按钮
        ttk.Button(frame_top, text="开始检测", style='success.TButton',
                   command=self.start_detection).grid(
            row=3, column=1, pady=10)

        ttk.Separator(self.root).pack(fill="x", padx=10, pady=10)

        # 下方控件
        frame_bottom = ttk.Frame(self.root)
        frame_bottom.pack(fill="x", padx=10, pady=10)

        # 训练配置文件
        ttk.Label(frame_bottom, text="训练配置文件 (data.yaml):").grid(
            row=0, column=0, sticky=W)
        ttk.Entry(frame_bottom, textvariable=self.yaml_path, width=50).grid(
            row=0, column=1, padx=5, sticky=EW)
        ttk.Button(frame_bottom, text="选择", command=self.browse_yaml).grid(
            row=0, column=2, padx=5)

        # 开始训练按钮
        ttk.Button(frame_bottom, text="开始训练", style='primary.TButton',
                   command=self.start_training).grid(
            row=1, column=1, pady=10)

        # 配置列权重使Entry可以扩展
        frame_top.columnconfigure(1, weight=1)
        frame_bottom.columnconfigure(1, weight=1)

        # 图片显示区域
        image_frame = ttk.Frame(self.root)
        image_frame.pack(fill= "both", expand=True, padx=10, pady=10)

        # 原图画布 - 确保使用 self. 保存为成员变量
        self.original_canvas = tk.Canvas(image_frame, relief="groove", bg='white')
        self.original_canvas.pack(side="left", fill= "both", expand=True, padx=10)
        self.original_canvas.create_text(150, 150, text="原图")  # 居中文本

        # 结果画布 - 确保使用 self. 保存为成员变量
        self.result_canvas = tk.Canvas(image_frame, relief="groove", bg='white')
        self.result_canvas.pack(side="right", fill= "both", expand=True, padx=10)
        self.result_canvas.create_text(150, 150, text="检测结果")  # 居中文本
    def browse_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch 模型", "*.pt")])
        if file_path:
            self.model_path.set(file_path)

    def browse_source(self):
        file_path = filedialog.askopenfilename(filetypes=[("媒体文件", "*.jpg *.png *.mp4")])
        if file_path:
            self.source_path.set(file_path)
            self.display_image(self.original_canvas, file_path)

    def browse_yaml(self):
        file_path = filedialog.askopenfilename(filetypes=[("YAML 文件", "*.yaml")])
        if file_path:
            self.yaml_path.set(file_path)

    def browse_save_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.save_dir.set(directory)

    def display_image(self, canvas, image_path):
        try:
            # 防御性检查
            if canvas is None:
                raise ValueError("Canvas 对象未初始化")

            # 检查图像路径有效性
            if not image_path or not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")

            # 清空画布前再次检查
            if hasattr(canvas, 'delete'):
                canvas.delete("all")  # 清空画布
            else:
                raise AttributeError("传入的对象不是有效的 Canvas")

            # 加载图像
            if image_path.lower().endswith((".jpg", ".jpeg", ".png")):
                img = Image.open(image_path)
            elif image_path.lower().endswith(".mp4"):
                cap = cv2.VideoCapture(image_path)
                success, frame = cap.read()
                cap.release()
                if not success:
                    return
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return

            # 自适应缩放
            canvas_width = canvas.winfo_width() or 400  # 默认值
            canvas_height = canvas.winfo_height() or 300

            # 保持宽高比缩放
            img.thumbnail((canvas_width, canvas_height))

            # 转换为 PhotoImage
            photo = ImageTk.PhotoImage(img)

            # 显示图像并保持引用
            canvas.create_image(canvas_width // 2, canvas_height // 2,
                                anchor="center", image=photo)
            canvas.image = photo  # 防止被垃圾回收

        except Exception as e:
            messagebox.showerror("显示错误", f"无法显示图像:\n{str(e)}")
    def start_detection(self):
        def detect():
            try:
                model = YOLO(self.model_path.get())
                results = model.predict(
                    source=self.source_path.get(),
                    save=True,
                    save_dir=self.save_dir.get() or None,
                    show=False  # 不弹出窗口，改为 GUI 中显示
                )

                # 获取检测后的图片路径
                result_img_path = Path(results[0].save_dir) / Path(results[0].path).name
                self.display_image(self.result_canvas, str(result_img_path))
                messagebox.showinfo("检测完成", f"检测完成，结果已保存到:\n{results[0].save_dir}")
            except Exception as e:
                messagebox.showerror("检测失败", str(e))

        threading.Thread(target=detect).start()

    def start_training(self):
        def train():
            try:
                model = YOLO(model=r'D:\yolo\yolov12\ultralytics\cfg\models\v12\yolov12.yaml')
                model.train(data=self.yaml_path.get(),
                            imgsz=640,
                            epochs=50,
                            batch=4,
                            workers=0,
                            device='',
                            optimizer='SGD',
                            close_mosaic=10,
                            resume=False,
                            project='runs/train',
                            name='exp',
                            single_cls=False,
                            cache=False)
                messagebox.showinfo("训练完成", "训练已完成。")
            except Exception as e:
                messagebox.showerror("训练失败", str(e))

        threading.Thread(target=train).start()


if __name__ == '__main__':
    root = tk.Tk()
    app = YOLOv12App(root)
    root.mainloop()
