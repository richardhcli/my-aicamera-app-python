#code to convert the YOLO model to a smaller format that can ru 
#more effectively on raspberry pi

from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model_name = "yolov8n.pt"#"yolo11n.pt"
model = YOLO(model_name)

# Export the model to NCNN format
model.export(format="ncnn")  # creates 'yolov8n_ncnn_model'

