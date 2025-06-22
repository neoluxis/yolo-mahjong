#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @file: mahjong_detect.py
# @brief: YOLOv8 Mahjong Detection using ONNX Runtime
# @author: Neolux
# @date: 2025-06-22

import argparse
import cv2 as cv
import numpy as np
import os
import onnxruntime as ort


class MahjongItem:
    """
    麻将数据
    """

    def __init__(self, class_id, box, score, classname):
        self.class_id = class_id
        self.box = box  # [left, top, width, height]
        self.score = score
        self.classname = classname

    def __repr__(self):
        return f"MahjongItem(class_id={self.class_id}, classname='{self.classname}')"


class YOLOv8:
    def __init__(
        self,
        model_path,
        input_size=(672, 672),
        conf_threshold=0.1,
        rms_threshold=0.1,
        device="cpu",
        classes="./classes.txt",
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.img_height, self.img_width = input_size
        self.conf_threshold = conf_threshold
        self.rms_threshold = rms_threshold

        if isinstance(classes, str):
            if not os.path.exists(classes):
                raise ValueError(f"Classes file does not exist: {classes}")
            with open(classes, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
        elif isinstance(classes, list):
            self.classes = classes

        if device == "cpu":
            self.provider = ["CPUExecutionProvider"]
        elif device == "cuda":
            self.provider = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            raise ValueError("Device must be 'cpu' or 'cuda'")
        self.session = ort.InferenceSession(self.model_path, providers=self.provider)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.dummy_input = {
            self.input_name: np.zeros((1, 3, *self.input_size), dtype=np.float32)
        }

    def letterbox(self, img, new_shape=(672, 672)):
        shape = img.shape[:2]

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv.copyMakeBorder(
            img, top, bottom, left, right, cv.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return img, (top, left)

    def preprocess(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image, pad = self.letterbox(image, new_shape=self.input_size)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image, pad

    def postprocess(
        self, output, pad, request=["boxes", "scores", "class_ids", "class_names"]
    ):
        outputs = np.transpose(np.squeeze(output[0]))

        rows = outputs.shape[0]

        boxes = []
        scores = []
        class_ids = []

        gain = min(
            self.input_size[0] / self.img_height, self.input_size[1] / self.img_width
        )
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        for i in range(rows):
            classes_scores = outputs[i][4:]

            max_score = np.amax(classes_scores)

            if max_score >= self.conf_threshold:
                class_id = np.argmax(classes_scores)

                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)

                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        indices = cv.dnn.NMSBoxes(
            boxes, scores, self.conf_threshold, self.rms_threshold
        )

        mahjong = []
        for i in indices:
            if i < len(boxes) and i < len(scores) and i < len(class_ids):
                class_id = class_ids[i]
                box = boxes[i]
                score = scores[i]
                classname = (
                    self.classes[class_id]
                    if class_id < len(self.classes)
                    else "Unknown"
                )
                mahjong.append(MahjongItem(class_id, box, score, classname))
        return mahjong

    def __call__(self, img):
        self.img_height, self.img_width = img.shape[:2]
        if self.img_height == 0 or self.img_width == 0:
            raise ValueError("Input image has invalid dimensions")
        input_image, pad = self.preprocess(img)
        output = self.session.run(self.output_names, {self.input_name: input_image})
        data = self.postprocess(output, pad)
        return data


class MahjongDetector:
    def __init__(self, model_path, device="cpu", classes="./classes.txt"):
        self.yolo = YOLOv8(model_path=model_path, device=device, classes=classes, conf_threshold=0.8)
        self.data_of_last = None

    def detect(self, image):
        data = self.yolo(image)
        data = sorted(data, key=lambda x: x.score, reverse=True)
        self.data_of_last = data
        return data


def main(args):
    yolo = YOLOv8(
        model_path=args.model,
        device=args.device,
        classes=args.classes,
        conf_threshold=0.8,
    )

    if os.path.isdir(args.image):
        images = [f"{args.image}/{x}" for x in os.listdir(args.image)]
    else:
        images = [args.image]
    images = sorted(images)

    mahjongs = []
    for image in images:
        img = cv.imread(image)
        if img is None:
            raise ValueError(f"Could not read image: {args.image}")

        data = yolo(img)
        print(f"{data=}")

        data = sorted(data, key=lambda x: x.score, reverse=True)
        for item in data:
            box = item.box
            class_id = item.class_id
            score = item.score
            classname = item.classname

            left, top, width, height = box
            right, bottom = left + width, top + height

            cv.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(
                img,
                f"Class: {class_id}, Score: {score:.2f}",
                (left, top - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        cv.imshow("Detection", img)
        cv.waitKey(0)
        mahjongs.append(data[0]) if len(data) > 0 else None
    print(f"{mahjongs=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Detection")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/mahjong/train/weights/best.onnx",
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="/home/neolux/workspace/SmartMahjong/images_grey/",
        help="Path to the input image or folder",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run the model on",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="./classes.txt",
        help="Path to the classes file or a list of class names",
    )

    args = parser.parse_args()

    main(args)
