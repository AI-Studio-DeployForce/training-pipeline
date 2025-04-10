# train_yolov9_clearml.py

from ultralytics import YOLO
from clearml import Task


# Initialize ClearML task (this will automatically log metrics, parameters, and artifacts)
task = Task.init(project_name='YOLOv9_Best_Training', task_name='Train_Best_YOLOv9')

# Define training parameters
data_config = './data.yaml'  # YAML file with dataset configuration (train/val paths, classes, etc.)
model_config = 'best.pt'                # Your YOLOv9 configuration file (ensure this file exists or modify accordingly)
epochs = 150                                 # Number of training epochs
img_size = 256                              # Image size (can adjust based on your requirements)

# Create the YOLO model (this uses the configuration file provided)
model = YOLO(model_config)

# Start training with the defined parameters.
# Ultralytics will automatically log training progress, and ClearML will capture these metrics.
results = model.train(data=data_config,
                        epochs=epochs,
                        imgsz=img_size,
                        patience=20,
                        batch=16,
                        lr0=0.00332871212734676,          # initial learning rate
                        lrf=0.287516089357777,          # final OneCycleLR learning rate
                        momentum=0.8582881626579386,    # SGD momentum / Adam beta1
                        weight_decay=0.0007277833518755533,   # optimizer weight decay
                        warmup_epochs=2.2333457665920773 ,     # number of warmup epochs
                        warmup_momentum=0.6087506436203991 ,   # initial momentum during warmup
                        warmup_bias_lr=0.08636367443379576 ,    # initial bias lr during warmup
                        box=7.5,           # box loss gain
                        cls=0.2518119276804113,           # classification loss gain
                        dfl=1.5,           # distribution focal loss gain
                        iou=0.451465626598339,        # IoU training threshold
                        hsv_h=0.0112806221363446 ,       # HSV hue augmentation
                        hsv_s=0.477413745349923,         # HSV saturation augmentation
                        hsv_v=0.4530873655047611 ,         # HSV value augmentation
                        degrees=13.593310551962048,       # rotation (+/- deg)
                        translate=0.08731078310093478,     # translation (+/- fraction)
                        scale=0.23062493349069543,         # scale (+/- gain)
                        shear=1.874164146170355,         # shear (+/- deg)
                        perspective=0.0005915287245521933,   # perspective (+/- fraction)
                        flipud=0.0012854999248946841,        # up-down flip probability
                        fliplr=0.4730939287124042,        # left-right flip probability
                        mosaic=0.772307690894658,        # mosaic augmentation probability
                        mixup=0.1032447247550029,        # mixup augmentation probability
                        copy_paste=0.14280887201059217,     # segment copy-paste augmentation probability
                        optimizer='AdamW',   # optimizer
)

# Optionally, you can log additional metrics or artifacts using ClearML APIs
# For example, logging the training results summary:
task.get_logger().report_text("Training completed successfully!")
print("Training completed!")


