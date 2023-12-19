from pathlib import Path
import PyQt5
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
from IPython.display import Markdown, display
from PIL import Image
from notebook_utils import load_image


def multiply_by_ratio(ratio_x, ratio_y, box):
    return [
        max(shape * ratio_y, 10) if idx % 2 else shape * ratio_x
        for idx, shape in enumerate(box[:-1])
    ]


def run_preprocesing_on_crop(crop, net_shape):
    temp_img = cv2.resize(crop, net_shape)
    temp_img = temp_img.reshape((1,) * 2 + temp_img.shape)
    return temp_img


def convert_result_to_image(bgr_image, resized_image, boxes, threshold=0.2, conf_labels=True):
    # RGB값의 코드 가독성을 높이기 위한 dictionary 선언
    colors = {"red": (255, 0, 0), "green": (0, 255, 0), "white": (255, 255, 255)}

    # resize 비율 계산
    (real_y, real_x), (resized_y, resized_x) = bgr_image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    # 이미지를 출력하기 위한 바탕 이미지 변수 선언
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    for box, annotation in boxes:
        conf = box[-1] # confidence값은 모델 output의 마지막 요소값이다.

        if conf > threshold:             
            (x_min, y_min, x_max, y_max) = map(int, multiply_by_ratio(ratio_x, ratio_y, box))
            cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["green"], 3) 

            if conf_labels: 
                cv2.putText(
                    rgb_image,
                    f"{annotation}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    colors["red"],
                    1,
                    cv2.LINE_AA,
                )

    return rgb_image



# 모델 준비
model_dir = Path("model")
precision = "FP16"
detection_model = "horizontal-text-detection-0001"
recognition_model = "text-recognition-resnet-fc"
detection_model_path = (model_dir / "intel/horizontal-text-detection-0001" / precision / detection_model).with_suffix(".xml")
recognition_model_path = (model_dir / "public/text-recognition-resnet-fc" / precision / recognition_model).with_suffix(".xml")

# 텍스트 detection 모델 생성
core = ov.Core()
detection_model = core.read_model(model=detection_model_path, weights=detection_model_path.with_suffix(".bin"))
detection_compiled_model = core.compile_model(model=detection_model, device_name='AUTO')

# 텍스트 detection 모델의 input shape에 맞게 이미지 transform 준비
detection_input_layer = detection_compiled_model.input(0)
N, C, H, W = detection_input_layer.shape
image = cv2.imread("sample_letter.jpg")
resized_image = cv2.resize(image, (W, H))
(real_y, real_x), (resized_y, resized_x) = image.shape[:2], resized_image.shape[:2]
ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

# 텍스트 detection 모델 추론
input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
output_key = detection_compiled_model.output("boxes")
boxes = detection_compiled_model([input_image])[output_key]
boxes = boxes[~np.all(boxes == 0, axis=1)] # 내용이 0뿐인 박스는 제거


# 텍스트 recognition 모델 생성
recognition_model = core.read_model(model=recognition_model_path, weights=recognition_model_path.with_suffix(".bin"))
recognition_compiled_model = core.compile_model(model=recognition_model, device_name='AUTO')

# 텍스트 recognition 모델의 input shape에 맞게 이미지 transform 준비
recognition_output_layer = recognition_compiled_model.output(0)
recognition_input_layer = recognition_compiled_model.input(0)
_, _, H, W = recognition_input_layer.shape


grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# "text-recognition-resnet-fc" 모델이 제공하는 encode output 문자열은 다음과 같음(나머지 문자열은 지원하지 않음)
letters = "~0123456789abcdefghijklmnopqrstuvwxyz"

# Prepare an empty list for annotations.
annotations = list()
cropped_images = list()
# fig, ax = plt.subplots(len(boxes), 1, figsize=(5,15), sharex=True, sharey=True)
# Get annotations for each crop, based on boxes given by the detection model.

for i, crop in enumerate(boxes):
    # Get coordinates on corners of a crop.
    (x_min, y_min, x_max, y_max) = map(int, multiply_by_ratio(ratio_x, ratio_y, crop))
    image_crop = run_preprocesing_on_crop(grayscale_image[y_min:y_max, x_min:x_max], (W, H))

    # Run inference with the recognition model.
    result = recognition_compiled_model([image_crop])[recognition_output_layer]

    # Squeeze the output to remove unnecessary dimension.
    recognition_results_test = np.squeeze(result)

    # Read an annotation based on probabilities from the output layer.
    annotation = list()
    for letter in recognition_results_test:
        parsed_letter = letters[letter.argmax()]

        # Returning 0 index from `argmax` signalizes an end of a string.
        if parsed_letter == letters[0]:
            break
        annotation.append(parsed_letter)
    annotations.append("".join(annotation))
    cropped_image = Image.fromarray(image[y_min:y_max, x_min:x_max])
    cropped_images.append(cropped_image)

boxes_with_annotations = list(zip(boxes, annotations))




plt.figure(figsize=(12, 12))
plt.imshow(convert_result_to_image(image, resized_image, boxes_with_annotations, conf_labels=True))


ocr_text_result = [
    annotation
    for _, annotation in sorted(zip(boxes, annotations), key=lambda x: x[0][0] ** 2 + x[0][1] ** 2)
]

# ocr_text_result = [
#     annotation
#     for _, annotation in zip(boxes, annotations)
# ]

# temp = [
#     box
#     for box, _ in zip(boxes, annotations)
# ]


# resultresultresult = list()
# for box, annotation in zip(boxes, annotations):
#     (x_min, y_min, x_max, y_max) = box[:-1]
#     resultresultresult.append((int(x_min), int(y_min), annotation))

# sorted_resultresultresult = sorted(resultresultresult, key=lambda x: (x[1], x[0]))  # 두 번째 요소를 기준으로 정렬
# print(sorted_resultresultresult)

# final_result = [
#     annotation 
#     for _, _, annotation in sorted_resultresultresult
# ]

# print("으악!!!")
print(ocr_text_result)
# print(temp)

result_string = ' '.join(ocr_text_result)
print(result_string)




