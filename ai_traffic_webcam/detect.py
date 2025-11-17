from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(
    source=0,       # webcam
    show=True,      # show live window
    conf=0.3
)
