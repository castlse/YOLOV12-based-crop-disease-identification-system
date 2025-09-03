# -*- coding: utf-8 -*-
# -*- coding: gbk -*-
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

class DiseaseDetectionApp:
    def __init__(self, root, model_path="yolov12n.pt"):
        self.root = root
        self.root.title("玉米病害检测系统")
        self.root.geometry("900x700")
        
        # 初始化模型
        self.model = YOLO(model_path)
        self.current_path = None
        self.is_video = False
        self.detection_results = None
        
        # 病害标签映射
        self.label_map = {
            0: "大斑病",
            1: "小斑病",
            2: "锈病",
            3: "灰斑病",
            4: "健康"
        }
        
        # 创建界面
        self._create_widgets()
        
    def _create_widgets(self):
        """创建界面组件"""
        # 顶部菜单栏
        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="选择模型", command=self._select_model)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menu_bar.add_cascade(label="文件", menu=file_menu)
        self.root.config(menu=menu_bar)
        
        # 主框架
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        control_frame = tk.LabelFrame(main_frame, text="控制区", padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        # 文件选择
        tk.Label(control_frame, text="选择图片/视频:").pack(anchor=tk.W, pady=5)
        
        select_frame = tk.Frame(control_frame)
        select_frame.pack(fill=tk.X, pady=5)
        
        self.file_path_var = tk.StringVar()
        self.file_path_entry = tk.Entry(select_frame, textvariable=self.file_path_var, width=30)
        self.file_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.select_btn = tk.Button(select_frame, text="浏览...", command=self._select_file)
        self.select_btn.pack(side=tk.RIGHT)
        
        # 开始检测按钮
        self.detect_btn = tk.Button(
            control_frame, text="开始检测", command=self._run_detection,
            state=tk.DISABLED, font=("SimHei", 12), bg="#4CAF50", fg="white"
        )
        self.detect_btn.pack(fill=tk.X, pady=20)
        
        # 右侧显示区域
        display_frame = tk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 预览区
        preview_frame = tk.LabelFrame(display_frame, text="预览区", padx=5, pady=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.preview_canvas = tk.Canvas(preview_frame, bg="#f0f0f0")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 结果区
        result_frame = tk.LabelFrame(display_frame, text="检测结果", padx=5, pady=5)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.result_text = tk.Text(result_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        self.status_bar = tk.Label(self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def _select_model(self):
        """选择自定义模型文件"""
        model_path = filedialog.askopenfilename(
            filetypes=[("YOLO模型", "*.pt"), ("所有文件", "*.*")]
        )
        
        if model_path:
            try:
                self.model = YOLO(model_path)
                messagebox.showinfo("成功", f"已加载模型: {model_path}")
            except Exception as e:
                messagebox.showerror("错误", f"加载模型失败: {str(e)}")
                
    def _select_file(self):
        """选择图片或视频文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp"), 
                       ("视频文件", "*.mp4;*.avi;*.mov;*.mkv"),
                       ("所有文件", "*.*")]
        )
        
        if file_path:
            self.current_path = file_path
            self.file_path_var.set(os.path.basename(file_path))
            self.is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
            
            # 更新UI状态
            self.detect_btn.config(state=tk.NORMAL)
            
            # 显示预览
            self._display_preview()


    def display_image(self, canvas, image_path):
        try:
            # 视频文件处理
            if image_path.endswith(".mp4"):
                cap = cv2.VideoCapture(image_path)
                success, frame = cap.read()
                cap.release()
                if not success or frame is None:
                    raise ValueError("无法读取视频帧")
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 图片文件处理
            elif image_path.endswith((".jpg", ".png", ".jpeg")):
                img = Image.open(image_path)

            else:
                raise ValueError("不支持的文件格式")

            # 图像缩放 & 显示
            img = img.resize((400, 300))
            photo = ImageTk.PhotoImage(img)
            canvas.config(image=photo)
            canvas.image = photo  # 必须保存引用，否则图像显示不出来

        except Exception as e:
            messagebox.showerror("图像显示错误", f"无法显示图像：{e}")

    def _run_detection(self):
        """执行病害检测"""
        if not self.current_path:
            messagebox.showwarning("警告", "请先选择图片或视频")
            return
            
        # 禁用按钮
        self.detect_btn.config(state=tk.DISABLED)
        self.status_bar.config(text="正在检测...")
        self.root.update()
        
        try:
            # 清空结果
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.config(state=tk.DISABLED)
            
            if self.is_video:
                # 视频检测
                self._detect_video()
            else:
                # 图片检测
                self._detect_image()
                
        except Exception as e:
            messagebox.showerror("错误", f"检测过程中出错: {str(e)}")
            self.status_bar.config(text="检测失败")
        finally:
            # 启用按钮
            self.detect_btn.config(state=tk.NORMAL)
            
    def _detect_image(self):
        """检测单张图片"""
        # 模型推理
        results = self.model(self.current_path, conf=0.5)
        
        # 解析检测结果
        self.detection_results = []
        for box in results[0].boxes:
            # 获取边界框坐标
            bbox = box.xyxy[0].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
            
            # 获取类别索引和置信度
            class_idx = int(box.cls)
            confidence = float(box.conf)
            
            # 添加到结果列表
            self.detection_results.append({
                "label": self.label_map.get(class_idx, f"未知类别{class_idx}"),
                "confidence": confidence,
                "bbox": tuple(bbox)
            })
            
        # 显示结果
        self._display_results()
        
        # 在原图上绘制边界框并更新预览
        if self.detection_results:
            image = Image.open(self.current_path)
            image_with_boxes = self._draw_boxes_on_image(image, self.detection_results)
            self._display_image_on_canvas(image_with_boxes)
            
    def _detect_video(self):
        """检测视频的第一帧"""
        # 只检测视频的第一帧作为示例
        cap = cv2.VideoCapture(self.current_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            messagebox.showerror("错误", "无法读取视频帧")
            return
            
        # 保存当前帧为临时文件
        temp_file = "temp_frame.jpg"
        cv2.imwrite(temp_file, frame)
        
        # 检测当前帧
        results = self.model(temp_file, conf=0.5)
        
        # 解析检测结果
        self.detection_results = []
        for box in results[0].boxes:
            # 获取边界框坐标
            bbox = box.xyxy[0].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
            
            # 获取类别索引和置信度
            class_idx = int(box.cls)
            confidence = float(box.conf)
            
            # 添加到结果列表
            self.detection_results.append({
                "label": self.label_map.get(class_idx, f"未知类别{class_idx}"),
                "confidence": confidence,
                "bbox": tuple(bbox)
            })
            
        # 删除临时文件
        os.remove(temp_file)
        
        # 显示结果
        self._display_results()
        
        # 在原图上绘制边界框并更新预览
        if self.detection_results:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            image_with_boxes = self._draw_boxes_on_image(image, self.detection_results)
            self._display_image_on_canvas(image_with_boxes)
            
    def _display_results(self):
        """显示检测结果"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        if not self.detection_results:
            self.result_text.insert(tk.END, "未检测到病害或检测失败\n")
            self.status_bar.config(text="检测完成: 未发现病害")
            self.result_text.config(state=tk.DISABLED)
            return
            
        # 统计各类病害数量
        disease_counts = {}
        for result in self.detection_results:
            label = result["label"]
            disease_counts[label] = disease_counts.get(label, 0) + 1
            
        # 显示统计结果
        self.result_text.insert(tk.END, "检测结果统计:\n")
        for label, count in disease_counts.items():
            self.result_text.insert(tk.END, f"- {label}: {count} 处\n")
            
        # 显示详细信息
        self.result_text.insert(tk.END, "\n详细信息:\n")
        for i, result in enumerate(self.detection_results, 1):
            self.result_text.insert(tk.END, f"\n病害 #{i}:\n")
            self.result_text.insert(tk.END, f"  类型: {result['label']}\n")
            self.result_text.insert(tk.END, f"  置信度: {result['confidence']*100:.1f}%\n")
            self.result_text.insert(tk.END, f"  位置: {result['bbox']}\n")
            
            # 添加防治建议
            self.result_text.insert(tk.END, "\n  防治建议:\n")
            self.result_text.insert(tk.END, f"  {self._get_treatment_advice(result['label'])}\n")
            
        self.status_bar.config(text="检测完成")
        self.result_text.config(state=tk.DISABLED)
        
    def _draw_boxes_on_image(self, image, results):
        """在图像上绘制边界框"""
        draw = ImageDraw.Draw(image)
        
        for result in results:
            label = result["label"]
            confidence = result["confidence"]
            bbox = result["bbox"]
            
            # 边界框颜色（健康为绿色，病害为红色）
            color = "green" if label == "健康" else "red"
            
            # 绘制边界框
            draw.rectangle(bbox, outline=color, width=2)
            
            # 绘制标签和置信度
            text = f"{label}: {confidence*100:.1f}%"
            draw.text((bbox[0], bbox[1]-15), text, fill=color)
            
        return image
        
    def _get_treatment_advice(self, disease_type):
        """获取病害防治建议"""
        advice_map = {
            "大斑病": "1. 及时清除病叶，集中烧毁或深埋\n2. 合理密植，保持田间通风透光\n3. 发病初期喷施多菌灵、百菌清等杀菌剂",
            "小斑病": "1. 选用抗病品种\n2. 加强田间管理，增施磷钾肥\n3. 发病时可喷施甲基托布津、代森锰锌等",
            "锈病": "1. 及时清除病残体\n2. 合理施肥，增强植株抗病性\n3. 发病初期喷施三唑酮、戊唑醇等药剂",
            "灰斑病": "1. 实行轮作倒茬\n2. 合理灌溉，降低田间湿度\n3. 发病初期喷施多菌灵、苯醚甲环唑等",
            "健康": "当前叶片健康，建议继续保持良好的田间管理，定期巡查，预防为主。"
        }
        
        return advice_map.get(disease_type, "请咨询农业专家获取针对性防治建议。")

if __name__ == "__main__":
    root = tk.Tk()
    app = DiseaseDetectionApp(root, model_path="yolov12.pt")  # 替换为你的模型路径
    root.mainloop()
