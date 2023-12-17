'''
Image classification using TeachableMachineLite and OpenCV example
    with Raspberry Pi and Camera Module 3.
By: RoboticX Team
'''
from teachable_machine_lite import TeachableMachineLite
import cv2 as cv

cap = cv.VideoCapture(0)

model_path = 'modelFiles/model.tflite'
image_file_name = "frame.jpg"
labels_path = "modelFiles/labels.txt"

tm_model = TeachableMachineLite(model_path=model_path, labels_file_path=labels_path)
frm_ctr = 0

prediction = ""
confidence = 0
while True:
    ret, frame = cap.read()
    if frm_ctr > 10:
        cv.imwrite(image_file_name, frame)        
        results = tm_model.classify_frame(image_file_name)
        # print("results:",results)
        print(results["label"])
        print(results["confidence"])
        prediction = results["label"]
        confidence = results["confidence"]
        frm_ctr=0
    else:
        frm_ctr+=1

    if confidence > 85.00:
        cv.putText(frame, prediction, (15,60), cv.FONT_ITALIC, 2, (0, 255, 0), 2)
    cv.imshow('Cam', frame)

    k = cv.waitKey(1)
    if k% 255 == 27:
        break