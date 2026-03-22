import os
import sys
from pathlib import Path

import torch
from einops import rearrange, repeat
from torchvision.ops import box_iou

CURRENT_DIR = Path(__file__).resolve().parent
ECCV22_ROOT = CURRENT_DIR / "eccv22"
MMDET_ROOT = CURRENT_DIR / "mmdetection"
YOLOV9_ROOT = CURRENT_DIR / "yolov9"

for path in (CURRENT_DIR, ECCV22_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class DifferenciableHumanDetector:
    def __init__(self):
        self._current_file_path = os.path.dirname(os.path.abspath(__file__))

    def _check_input(self, rgb, thermal):
        assert len(rgb.shape) == 4, f"RGB must have 4 dimensions, but got {rgb.shape}"
        assert len(thermal.shape) == 4, f"Thermal must have 4 dimensions, but got {thermal.shape}"
        assert rgb.shape[0] == thermal.shape[0], "Batch size of RGB and Thermal must be the same"
        assert rgb.shape[1] == 3, f"RGB must have 3 channels, but got {rgb.shape[1]}"
        assert thermal.shape[1] == 1, f"Thermal must have 1 channel, but got {thermal.shape[1]}"
        assert 0 <= rgb.min() and rgb.max() <= 1, f"RGB must be normalized to [0, 1], but got {rgb.min()} and {rgb.max()}"
        assert 0 <= thermal.min() and thermal.max() <= 1, f"Thermal must be normalized to [0, 1], but got {thermal.min()} and {thermal.max()}"

    def _check_boxes(self, boxes):
        assert len(boxes.shape) == 2, f"Boxes must have 2 dimensions, but got {boxes.shape}"
        assert boxes.shape[1] == 4, f"Boxes must have 4 coordinates, but got {boxes.shape[1]}"
        assert (boxes[:, 0] >= boxes[:, 2]).all() or (boxes[:, 0] <= boxes[:, 2]).all(), f"Boxes must have x1 <= x2, but got {boxes[:, 0]} and {boxes[:, 2]}"
        assert (boxes[:, 1] >= boxes[:, 3]).all() or (boxes[:, 1] <= boxes[:, 3]).all(), f"Boxes must have y1 <= y2, but got {boxes[:, 1]} and {boxes[:, 3]}"
        assert not boxes.max() <= 1, f"Boxes should not be normalized to [0, 1], but got {boxes.max()}"

    def detect(self, rgb: torch.Tensor, thermal: torch.Tensor):
        raise NotImplementedError

    def boxed_detect(self, rgb: torch.Tensor, thermal: torch.Tensor, boxes: torch.Tensor, iou_thr):
        raise NotImplementedError


def _filter_person_instances(output, device):
    if len(output["instances"]) == 0:
        return torch.empty((0, 4), device=device), torch.empty((0,), device=device)

    pred_classes = output["instances"].pred_classes
    boxes = output["instances"].pred_boxes.tensor
    scores = output["instances"].scores
    mask = pred_classes == 0
    return boxes[mask], scores[mask]


def _max_score_for_boxes(outputs, boxes, iou_thr, device):
    max_scores = []
    for index, output in enumerate(outputs):
        if len(output["instances"]) == 0:
            max_scores.append(torch.tensor(0.0, device=device))
            continue

        pred_boxes = output["instances"].pred_boxes
        target_box = boxes[index].unsqueeze(0).to(pred_boxes.device)
        ious = box_iou(target_box, pred_boxes.tensor)[0]
        scores = output["instances"].scores
        pred_classes = output["instances"].pred_classes
        mask = (pred_classes == 0) & (ious >= iou_thr)
        filtered_scores = scores[mask]
        if len(filtered_scores) > 0:
            max_scores.append(filtered_scores.max())
        else:
            max_scores.append(torch.tensor(0.0, device=device))

    return torch.stack(max_scores).mean()


class ECCV22EarlyFusionDetector(DifferenciableHumanDetector):
    def __init__(self, training=False):
        super().__init__()
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(self._current_file_path, "eccv22/configs/FLIR_early_fusion_config.yaml"))
        cfg = self.init_config(cfg)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01 if training else 0.5
        self.model = DefaultPredictor(cfg).model

    def init_config(self, cfg):
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.INPUT.FORMAT = "BGRT"
        cfg.INPUT.NUM_IN_CHANNELS = 4
        cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438]
        cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
        cfg.MODEL.WEIGHTS = os.path.join(self._current_file_path, "../assets/detection_ckpt/FLIR_early_fusion.pth")
        return cfg

    def detect(self, rgb: torch.Tensor, thermal: torch.Tensor):
        self._check_input(rgb, thermal)
        bgrt = 255.0 * torch.cat([rgb[:, [2, 1, 0], ...].clone(), thermal.clone()], dim=1)
        height, width = bgrt.shape[2:]
        inputs = [{"image": bgrt[i], "height": height, "width": width} for i in range(bgrt.shape[0])]
        return self.model(inputs)

    def display_detect(self, rgb: torch.Tensor, thermal: torch.Tensor):
        outputs = self.detect(rgb, thermal)
        batch_boxes = []
        batch_scores = []
        for output in outputs:
            boxes, scores = _filter_person_instances(output, self.model.device)
            batch_boxes.append(boxes)
            batch_scores.append(scores)
        return batch_boxes, batch_scores

    def boxed_detect(self, rgb: torch.Tensor, thermal: torch.Tensor, boxes: torch.Tensor, iou_thr):
        self._check_boxes(boxes)
        outputs = self.detect(rgb, thermal)
        return _max_score_for_boxes(outputs, boxes, iou_thr, self.model.device)


class ECCV22MiddleFusionDetector(DifferenciableHumanDetector):
    def __init__(self, training=False):
        super().__init__()
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(self._current_file_path, "eccv22/configs/FLIR_mid_fusion_config.yaml"))
        cfg = self.init_config(cfg)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01 if training else 0.5
        self.model = DefaultPredictor(cfg).model

    def init_config(self, cfg):
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.INPUT.FORMAT = "BGRTTT"
        cfg.INPUT.NUM_IN_CHANNELS = 6
        cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438, 135.438, 135.438]
        cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        cfg.MODEL.WEIGHTS = os.path.join(self._current_file_path, "../assets/detection_ckpt/FLIR_middle_fusion.pth")
        return cfg

    def detect(self, rgb: torch.Tensor, thermal: torch.Tensor):
        self._check_input(rgb, thermal)
        rgbttt = 255.0 * torch.cat([rgb, thermal.repeat(1, 3, 1, 1)], dim=1)
        height, width = rgbttt.shape[2:]
        inputs = [{"image": rgbttt[i], "height": height, "width": width} for i in range(rgbttt.shape[0])]
        return self.model(inputs)

    def display_detect(self, rgb: torch.Tensor, thermal: torch.Tensor):
        outputs = self.detect(rgb, thermal)
        batch_boxes = []
        batch_scores = []
        for output in outputs:
            boxes, scores = _filter_person_instances(output, self.model.device)
            batch_boxes.append(boxes)
            batch_scores.append(scores)
        return batch_boxes, batch_scores

    def boxed_detect(self, rgb: torch.Tensor, thermal: torch.Tensor, boxes: torch.Tensor, iou_thr):
        self._check_boxes(boxes)
        outputs = self.detect(rgb, thermal)
        return _max_score_for_boxes(outputs, boxes, iou_thr, self.model.device)


class ECCV22ThermalOnlyDetector(DifferenciableHumanDetector):
    def __init__(self, training=False):
        super().__init__()
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(self._current_file_path, "eccv22/configs/FLIR_thermal_only_config.yaml"))
        cfg = self.init_config(cfg)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01 if training else 0.5
        self.model = DefaultPredictor(cfg).model

    def init_config(self, cfg):
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.WEIGHTS = os.path.join(self._current_file_path, "../assets/detection_ckpt/FLIR_late_fusion_thermal.pth")
        return cfg

    def detect(self, rgb: torch.Tensor, thermal: torch.Tensor):
        self._check_input(rgb, thermal)
        thermal_input = 255.0 * thermal
        height, width = thermal_input.shape[2:]
        inputs = [{"image": thermal_input[i, 0], "height": height, "width": width} for i in range(thermal_input.shape[0])]
        return self.model(inputs)

    def display_detect(self, rgb: torch.Tensor, thermal: torch.Tensor):
        outputs = self.detect(rgb, thermal)
        batch_boxes = []
        batch_scores = []
        for output in outputs:
            boxes, scores = _filter_person_instances(output, self.model.device)
            batch_boxes.append(boxes)
            batch_scores.append(scores)
        return batch_boxes, batch_scores

    def boxed_detect(self, rgb: torch.Tensor, thermal: torch.Tensor, boxes: torch.Tensor, iou_thr):
        self._check_boxes(boxes)
        outputs = self.detect(rgb, thermal)
        return _max_score_for_boxes(outputs, boxes, iou_thr, self.model.device)


class YOLOvXDetector(DifferenciableHumanDetector):
    def __init__(self, training=False, version="yolo11x", mode="jointmax"):
        del training
        print(f"loading {version} detector ...")
        super().__init__()
        if not version.startswith("yolo11"):
            raise ValueError(f"Unsupported YOLO version for minimal repo: {version}")

        self.mode = mode
        self.checkpoint_path = os.path.join(self._current_file_path, f"../assets/detection_ckpt/{version}.pt")
        yolov9_root = str(YOLOV9_ROOT)
        if yolov9_root not in sys.path:
            sys.path.insert(0, yolov9_root)
        from models.common import DetectMultiBackend

        self.model = DetectMultiBackend(self.checkpoint_path, device=torch.device("cuda"))
        print(f"{version} detector loaded")

    def display_detect(self, rgb: torch.Tensor, thermal: torch.Tensor):
        self._check_input(rgb, thermal)
        thermal = repeat(thermal, "b 1 h w -> b 3 h w")
        results_rgb = self.model(rgb)[0]
        results_thermal = self.model(thermal)[0]
        if isinstance(results_rgb, list):
            results_rgb = torch.cat(results_rgb, dim=0)
        if isinstance(results_thermal, list):
            results_thermal = torch.cat(results_thermal, dim=0)
        results_rgb = rearrange(results_rgb, "b t n -> b n t")
        results_thermal = rearrange(results_thermal, "b t n -> b n t")

        def process_batch_results(batch_results):
            batch_boxes = []
            batch_scores = []
            for results in batch_results:
                class_pred = results[..., 4:].argmax(dim=-1)
                mask = class_pred == 0
                filtered_results = results[mask]
                if len(filtered_results) == 0:
                    batch_boxes.append(None)
                    batch_scores.append(None)
                    continue

                bboxes = filtered_results[:, :4]
                conf = filtered_results[:, 4]
                max_conf_idx = conf.argmax()
                x, y, w, h = bboxes[max_conf_idx]
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                batch_boxes.append(torch.tensor([[x1, y1, x2, y2]], device=bboxes.device))
                batch_scores.append(conf[max_conf_idx:max_conf_idx + 1])
            return batch_boxes, batch_scores

        rgb_boxes, rgb_scores = process_batch_results(results_rgb)
        thermal_boxes, thermal_scores = process_batch_results(results_thermal)
        final_boxes = []
        final_scores = []
        for index in range(len(rgb_boxes)):
            if rgb_boxes[index] is None and thermal_boxes[index] is None:
                final_boxes.append(None)
                final_scores.append(None)
            elif rgb_boxes[index] is None:
                final_boxes.append(thermal_boxes[index])
                final_scores.append(thermal_scores[index])
            elif thermal_boxes[index] is None:
                final_boxes.append(rgb_boxes[index])
                final_scores.append(rgb_scores[index])
            elif rgb_scores[index] > thermal_scores[index]:
                final_boxes.append(rgb_boxes[index])
                final_scores.append(rgb_scores[index])
            else:
                final_boxes.append(thermal_boxes[index])
                final_scores.append(thermal_scores[index])
        return final_boxes, final_scores

    def boxed_detect(self, rgb: torch.Tensor, thermal: torch.Tensor, boxes: torch.Tensor, iou_thr):
        self._check_input(rgb, thermal)
        self._check_boxes(boxes)
        thermal = repeat(thermal, "b 1 h w -> b 3 h w")
        results_rgb = self.model(rgb)[0]
        results_thermal = self.model(thermal)[0]
        if isinstance(results_rgb, list):
            results_rgb = torch.cat(results_rgb, dim=0)
        if isinstance(results_thermal, list):
            results_thermal = torch.cat(results_thermal, dim=0)
        results_rgb = rearrange(results_rgb, "b t n -> b n t")
        results_thermal = rearrange(results_thermal, "b t n -> b n t")

        very_small_number = torch.mean(results_rgb) * 0.0 + torch.mean(results_thermal) * 0.0

        def process_batch_results(batch_results, batch_boxes):
            batch_confs = []
            for results, target_box in zip(batch_results, batch_boxes):
                class_pred = results[..., 4:].argmax(dim=-1)
                mask = class_pred == 0
                filtered_results = results[mask]
                if len(filtered_results) == 0:
                    batch_confs.append(torch.tensor(very_small_number, device=results.device))
                    continue

                bboxes = filtered_results[:, :4]
                conf = filtered_results[:, 4]
                x = bboxes[:, 0]
                y = bboxes[:, 1]
                w = bboxes[:, 2]
                h = bboxes[:, 3]
                pred_boxes = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=1)
                ious = box_iou(pred_boxes, target_box.unsqueeze(0))[..., 0]
                valid_conf = conf[ious >= iou_thr]
                if len(valid_conf) == 0:
                    batch_confs.append(torch.tensor(very_small_number, device=results.device))
                else:
                    batch_confs.append(valid_conf.max())
            return torch.stack(batch_confs)

        rgb_confs = process_batch_results(results_rgb, boxes)
        thermal_confs = process_batch_results(results_thermal, boxes)
        if self.mode == "jointmax":
            max_confs = torch.maximum(rgb_confs, thermal_confs)
        elif self.mode == "jointmean":
            max_confs = (rgb_confs + thermal_confs) / 2
        elif self.mode == "rgb":
            max_confs = rgb_confs
        elif self.mode == "thermal":
            max_confs = thermal_confs
        else:
            raise ValueError(f"Unsupported YOLO fusion mode: {self.mode}")

        return max_confs.mean()


class FasterRCNNRGBDetector(DifferenciableHumanDetector):
    def __init__(self, training=False, mode="jointmax"):
        del training
        super().__init__()
        mmdet_root = str(MMDET_ROOT)
        if mmdet_root not in sys.path:
            sys.path.insert(0, mmdet_root)

        from mmdet.apis import init_detector
        from mmdet.structures import DetDataSample

        self._det_data_sample_cls = DetDataSample
        self.model = init_detector(
            os.path.join(self._current_file_path, "mmdetection/faster-rcnn_r50_fpn_1x_coco.py"),
            os.path.join(self._current_file_path, "../assets/detection_ckpt/FLIR_late_fusion_rgb.pth"),
            device="cuda",
        )
        self.mode = mode

    def _detect(self, rgb: torch.Tensor):
        rgb = rgb.cuda() * 255
        batch_size = rgb.shape[0]
        data_samples = [self._det_data_sample_cls() for _ in range(batch_size)]
        for index in range(batch_size):
            data_samples[index].set_metainfo(
                {
                    "img_shape": rgb[index].shape[1:],
                    "ori_shape": rgb[index].shape[1:],
                    "scale_factor": (1.0, 1.0),
                }
            )
        results_rgb = self.model.test_step(dict(inputs=rgb, data_samples=data_samples))
        processed_results = []
        for result in results_rgb:
            pred_instances = result.pred_instances
            processed_results.append(
                {
                    "boxes": pred_instances.bboxes,
                    "scores": pred_instances.scores,
                    "labels": pred_instances.labels,
                }
            )
        return processed_results

    def display_detect(self, rgb: torch.Tensor, thermal: torch.Tensor):
        self._check_input(rgb, thermal)
        processed_results = self._detect(rgb)
        final_boxes = []
        final_scores = []
        for result in processed_results:
            person_mask = result["labels"] == 0
            filtered_scores = result["scores"][person_mask]
            if len(filtered_scores) == 0:
                final_boxes.append(None)
                final_scores.append(None)
                continue
            score, box_index = torch.max(filtered_scores, dim=0)
            bbox = result["boxes"][person_mask][box_index.item()]
            final_boxes.append(bbox.unsqueeze(0))
            final_scores.append(score.unsqueeze(0))
        return final_boxes, final_scores

    def boxed_detect(self, rgb: torch.Tensor, thermal: torch.Tensor, boxes: torch.Tensor, iou_thr):
        self._check_input(rgb, thermal)
        self._check_boxes(boxes)
        processed_results = self._detect(rgb)
        final_scores = []
        very_small_number = torch.tensor(0.0, device=rgb.device)

        for index, result in enumerate(processed_results):
            gt_bbox = boxes[index].unsqueeze(0)
            person_mask = result["labels"] == 0
            if not person_mask.any():
                final_scores.append(very_small_number)
                continue

            pred_boxes = result["boxes"][person_mask]
            pred_scores = result["scores"][person_mask]
            ious = box_iou(pred_boxes, gt_bbox)
            valid_mask = ious.squeeze(1) >= iou_thr
            score = pred_scores[valid_mask].max() if valid_mask.any() else very_small_number
            final_scores.append(score)

        return torch.stack(final_scores).mean()


class ECCV22LateFusionDetector(DifferenciableHumanDetector):
    def __init__(self, training=False):
        super().__init__()
        self.thermal_detector = ECCV22ThermalOnlyDetector(training=training)
        self.rgb_detector = FasterRCNNRGBDetector(training=training)

    def display_detect(self, rgb: torch.Tensor, thermal: torch.Tensor):
        self._check_input(rgb, thermal)
        thermal_boxes, thermal_scores = self.thermal_detector.display_detect(rgb, thermal)
        rgb_boxes, rgb_scores = self.rgb_detector.display_detect(rgb, thermal)

        final_boxes = []
        final_scores = []
        for index in range(len(thermal_boxes)):
            rgb_valid = (
                rgb_boxes[index] is not None
                and isinstance(rgb_boxes[index], torch.Tensor)
                and rgb_boxes[index].numel() > 0
                and rgb_scores[index] is not None
            )
            thermal_valid = (
                thermal_boxes[index] is not None
                and isinstance(thermal_boxes[index], torch.Tensor)
                and thermal_boxes[index].numel() > 0
                and thermal_scores[index] is not None
            )

            if not thermal_valid and not rgb_valid:
                final_boxes.append(None)
                final_scores.append(None)
            elif not thermal_valid:
                final_boxes.append(rgb_boxes[index])
                final_scores.append(rgb_scores[index])
            elif not rgb_valid:
                final_boxes.append(thermal_boxes[index])
                final_scores.append(thermal_scores[index])
            elif thermal_scores[index].max().item() > rgb_scores[index].max().item():
                final_boxes.append(thermal_boxes[index])
                final_scores.append(thermal_scores[index])
            else:
                final_boxes.append(rgb_boxes[index])
                final_scores.append(rgb_scores[index])
        return final_boxes, final_scores

    def boxed_detect(self, rgb: torch.Tensor, thermal: torch.Tensor, boxes: torch.Tensor, iou_thr):
        self._check_input(rgb, thermal)
        self._check_boxes(boxes)
        thermal_score = self.thermal_detector.boxed_detect(rgb, thermal, boxes, iou_thr)
        rgb_score = self.rgb_detector.boxed_detect(rgb, thermal, boxes, iou_thr)
        return torch.maximum(thermal_score, rgb_score)
