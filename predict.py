from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'D:\yolo\yolov12\runs\train\exp7\weights\best.pt')
    model.predict(source=r'D:\yolo\yolov12\test.mp4',
                  save=True,
                  show=True,
                  )