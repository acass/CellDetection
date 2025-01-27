from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("cell.jpg", save=True, imgsz=320, conf=0.5)





# from ultralytics import YOLO
#
# # Load a model
# model = YOLO("best.pt")  # pretrained YOLO11n model
#
# # Run batched inference on a list of images
# results = model(["cell.png"], stream=True, save=True)
#
# # Process results generator
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk