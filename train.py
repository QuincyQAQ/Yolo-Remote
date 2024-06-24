from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/v8/Yolo-Remote.yaml")
    model.train(data="datasets/RSOD/RSOD.yaml")
