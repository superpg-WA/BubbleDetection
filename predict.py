from ultralytics import YOLO
import cv2

model_path = 'D:\\南京大学\\毕业设计\\labelme2YoloV8-segment\\runs\\segment\\train10\\weights\\best.pt'
img_path = 'D:\\南京大学\\毕业设计\\labelme2YoloV8-segment\\test\\GOLD_L1D_CHA_NI1_1356_2022_101_22_40_v05_r01_c05.png'
model = YOLO(model=model_path)

image = cv2.imread(img_path)
results = model.predict(image)

# # 把tensor转为numpy格式
# boxes = results[0].boxes.cpu().numpy()
#
# # 输出模型中有哪些类别
# print(results[0].names)
#
# print("berfore")
#
# # 访问 boxes 属性，它包含了检测到的边界框，对应的类别得分，及对应的类别
# loc, scores, classes = [], [], []
#
# # # 遍历每个检测结果
# for box in boxes:
#     loc.append(box.xyxy[0].tolist())
#     scores.append(float(box.conf))
#     classes.append(results[0].names[int(box.cls)])
#
# print(loc)
# print(scores)
# print(classes)

import numpy as np

# 遍历每个预测结果
for result in results:
    # 获取分割掩码
    masks = result.masks.data.numpy()
    # 获取类别标签
    class_ids = result.boxes.cls.numpy().astype(int)
    # 获取边界框坐标
    boxes = result.boxes.xyxy.numpy()

    # 遍历每个实例
    for mask, class_id, box in zip(masks, class_ids, boxes):
        # 将掩码转换为 uint8 类型
        mask = (mask * 255).astype(np.uint8)
        # 找到掩码中的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 绘制边界框
        # cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        # 在边界框上方绘制类别标签
        cv2.putText(image, model.names[class_id], (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # 绘制分割掩码轮廓
        cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

cv2.imshow('Instance Segmentation Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()