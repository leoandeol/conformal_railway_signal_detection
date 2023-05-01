import os
import pickle
import tqdm
import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt

import torchvision.transforms as T


from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg

from diffusiondet.predictor import VisualizationDemo
from diffusiondet import (
    DiffusionDetDatasetMapper,
    add_diffusiondet_config,
    DiffusionDetWithTTA,
)
from diffusiondet.util.model_ema import (
    add_model_ema_configs,
    may_build_model_ema,
    may_get_ema_checkpointer,
    EMAHook,
    apply_model_ema_and_restore,
    EMADetectionCheckpointer,
)

from confvision.data import plot_preds_img
from confvision.evaluate import (
    getAveragePrecision,
    get_recall_precision,
    plot_recall_precision,
    getCoverage,
    getRisk,
    getStretch,
)


def nms(pred_boxes, scores, iou_threshold=0.5):
    # TODO : unitary tests !
    new_boxes = []
    new_scores = []
    for boxes, score in zip(pred_boxes, scores):
        # If no bounding boxes, return empty list
        if len(boxes) == 0:
            new_boxes.append([])
            new_scores.append([])
            continue

        # Bounding boxes
        boxes = boxes.detach().cpu().numpy()

        # coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]

        # Confidence scores of bounding boxes
        score = score.detach().cpu().numpy()

        # Picked bounding boxes
        picked_boxes = []
        picked_score = []

        # Compute areas of bounding boxes
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)

        # Sort by confidence score of bounding boxes
        order = np.argsort(score)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            picked_boxes.append(boxes[index])
            picked_score.append(score[index])

            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
            intersection = w * h

            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

            left = np.where(ratio < iou_threshold)
            order = order[left]

        new_boxes.append(torch.FloatTensor(np.array(picked_boxes)).cuda())
        new_scores.append(torch.FloatTensor(np.array(picked_score)).cuda())
    return new_boxes, new_scores


class Predictions:
    def __init__(self, images, true_boxes, pred_boxes, scores, conf_boxes=None):
        self.images = images
        self.true_boxes = true_boxes
        self.pred_boxes = pred_boxes
        self.scores = scores
        self.conf_boxes = conf_boxes

        assert len(images) == len(true_boxes)
        assert len(images) == len(pred_boxes)

    def set_conf_boxes(self, conf_boxes):
        self.conf_boxes = conf_boxes
        self.AP_conf = None
        self.total_recalls_conf = None
        self.total_precisions_conf = None
        self.threshes_objectness_conf = None
        assert len(self.images) == len(conf_boxes)

    def plot_img(self, idx, score_threshold=0):
        plot_preds_img(
            idx, self, score_threshold=score_threshold, conf_boxes=self.conf_boxes
        )

    def plot_objectness_hist(self, **kwargs):
        objs = np.concatenate([x.detach().cpu().numpy() for x in self.scores])
        plt.hist(objs, **kwargs)
        plt.xlabel("Objectness score")
        plt.ylabel("Number of predicted boxes")
        plt.title("Histogram of objectness scores (log scale)")
        plt.show()

    def getRecallPrecision(self, iou_threshold, objectness_threshold, verbose=True):
        if self.conf_boxes is None:
            if verbose:
                print("Computing recall and precision on predicted boxes")
            recalls, precisions, scores = get_recall_precision(
                self.images,
                self.true_boxes,
                self.pred_boxes,
                self.scores,
                IOU_THRESHOLD=iou_threshold,
                SCORE_THRESHOLD=objectness_threshold,
                verbose=verbose,
            )
            return recalls, precisions, scores
        else:
            if verbose:
                print("Computing recall and precision on predicted boxes")
            recalls, precisions, scores = get_recall_precision(
                self.images,
                self.true_boxes,
                self.pred_boxes,
                self.scores,
                IOU_THRESHOLD=iou_threshold,
                SCORE_THRESHOLD=objectness_threshold,
                verbose=verbose,
            )
            if verbose:
                print("Computing recall and precision on conformalized boxes")
            recalls_conf, precisions_conf, scores_conf = get_recall_precision(
                self.images,
                self.true_boxes,
                self.conf_boxes,
                self.scores,
                IOU_THRESHOLD=iou_threshold,
                SCORE_THRESHOLD=objectness_threshold,
                verbose=verbose,
            )
            return (recalls, precisions, scores), (
                recalls_conf,
                precisions_conf,
                scores_conf,
            )

    def getAveragePrecision(self, verbose=True):
        # TODO: if called with same conf_boxes then just return the saved values directly
        if self.conf_boxes is None:
            if verbose:
                print("Computing average precision on predicted boxes")
            (
                AP,
                total_recalls,
                total_precisions,
                threshes_objectness,
            ) = getAveragePrecision(
                self.images,
                self.true_boxes,
                self.pred_boxes,
                self.scores,
                verbose=verbose,
            )
            self.AP = AP
            self.total_recalls = total_recalls
            self.total_precisions = total_precisions
            self.threshes_objectness = threshes_objectness
            if verbose:
                print("Average Precision on predictions =", AP)
            return AP, total_recalls, total_precisions, threshes_objectness
        else:
            if verbose:
                print("Computing average precision on predicted boxes")
            (
                AP,
                total_recalls,
                total_precisions,
                threshes_objectness,
            ) = getAveragePrecision(
                self.images,
                self.true_boxes,
                self.pred_boxes,
                self.scores,
                verbose=verbose,
            )
            self.AP = AP
            self.total_recalls = total_recalls
            self.total_precisions = total_precisions
            self.threshes_objectness = threshes_objectness
            if verbose:
                print("Computing average precision on conformalized boxes")
            (
                AP_conf,
                total_recalls_conf,
                total_precisions_conf,
                threshes_objectness_conf,
            ) = getAveragePrecision(
                self.images,
                self.true_boxes,
                self.conf_boxes,
                self.scores,
                verbose=verbose,
            )
            self.AP_conf = AP_conf
            self.total_recalls_conf = total_recalls_conf
            self.total_precisions_conf = total_precisions_conf
            self.threshes_objectness_conf = threshes_objectness_conf
            if verbose:
                print(
                    "Average Precision on predictions =",
                    AP,
                    ", on conformalized predictions =",
                    AP_conf,
                )
            return (AP, total_recalls, total_precisions, threshes_objectness), (
                AP_conf,
                total_recalls_conf,
                total_precisions_conf,
                threshes_objectness_conf,
            )

    def plot_recall_precision(self):
        out = self.getAveragePrecision(verbose=False)
        if len(out) == 2:
            (AP, total_recalls, total_precisions, threshes_objectness), (
                AP_conf,
                total_recalls_conf,
                total_precisions_conf,
                threshes_objectness_conf,
            ) = out
            plot_recall_precision(total_recalls, total_precisions, threshes_objectness)
            plot_recall_precision(
                total_recalls_conf, total_precisions_conf, threshes_objectness_conf
            )
        else:
            AP, total_recalls, total_precisions, threshes_objectness
            plot_recall_precision(total_recalls, total_precisions, threshes_objectness)

    def getRisk(self, loss="recall", objectness_threshold=0.3):
        if self.conf_boxes is None:
            risk = getRisk(
                self.images,
                self.true_boxes,
                self.pred_boxes,
                self.scores,
                loss=loss,
                objectness_threshold=objectness_threshold,
            )
            print(f"Average risk of predictions = {risk:.2f}")
            return risk
        else:
            risk, risk_conf = getRisk(
                self.images,
                self.true_boxes,
                self.pred_boxes,
                self.scores,
                loss=loss,
                objectness_threshold=objectness_threshold,
            ), getRisk(
                self.images,
                self.true_boxes,
                self.conf_boxes,
                self.scores,
                loss=loss,
                objectness_threshold=objectness_threshold,
            )
            print(
                f"Average risk of predictions = {risk:.2f} and of conformalized predictions = {risk_conf:.2f}"
            )
            return risk, risk_conf

    def getCoverage(self, objectness_threshold=0.5, iou_threshold=0.5):
        # TODO add print about pred and conf, also for getRisk
        if self.conf_boxes is None:
            coverage = getCoverage(
                self.images,
                self.true_boxes,
                self.pred_boxes,
                self.pred_boxes,
                self.scores,
                objectness_threshold=objectness_threshold,
                iou_threshold=iou_threshold,
            )
            print(f"Coverage of predictions = {coverage:.2f}")
            return coverage
        else:
            coverage, coverage_conf = getCoverage(
                self.images,
                self.true_boxes,
                self.pred_boxes,
                self.pred_boxes,
                self.scores,
                objectness_threshold=objectness_threshold,
                iou_threshold=iou_threshold,
            ), getCoverage(
                self.images,
                self.true_boxes,
                self.pred_boxes,
                self.conf_boxes,
                self.scores,
                objectness_threshold=objectness_threshold,
                iou_threshold=iou_threshold,
            )
            print(
                f"Coverage of predictions = {coverage:.2f} and of conformalized predictions = {coverage_conf:.2f}"
            )

            return coverage, coverage_conf

    def getStretch(self, verbose=True):
        if self.conf_boxes is not None:
            stretch = getStretch(self.pred_boxes, self.true_boxes)
            if verbose:
                print(f"Average stretch = {stretch}")
            return stretch
        else:
            if verbose:
                print("No conformalized boxes, cannot compute stretch")
            return None

    def describe(
        self,
        mode="classic",
        objectness_threshold=0.5,
        iou_threshold=0.5,
        loss="recall",
        verbose=False,
    ):
        # model : classic or risk
        if verbose:
            self.plot_objectness_hist(bins=20, log=True)
        self.getAveragePrecision(verbose=verbose)
        if mode == "classic":
            self.getCoverage(
                objectness_threshold=objectness_threshold,
                iou_threshold=iou_threshold,
            )
        elif mode == "risk":
            self.getRisk(objectness_threshold=objectness_threshold, loss=loss)
        print("Temporary removing strethc")  # self.getStretch()


class AbstractModel:
    def __init__(self):
        self._predictions = None

    def _get_predictions(self, dataloader, tqdm_on=True):
        dl_iter = iter(dataloader)
        nb_batches = len(dl_iter)
        images = []
        true_boxes = []
        pred_boxes = []
        pred_scores = []
        # predictor.input_format = 'BGR'
        # predictor.input_format = 'RGB' # default
        for i in tqdm.tqdm(range(nb_batches), disable=not tqdm_on):
            x, y = next(dl_iter)
            for xi, yi in zip(x, y):
                with torch.no_grad():
                    # [B, C, H, W]  B=1 pour l'instant
                    pred_boxs, pred_classes = self.predict(x)

                images.append(xi)
                true_boxes.append(yi)
                pred_boxes.append(pred_boxs)
                pred_scores.append(pred_classes)
        return images, true_boxes, pred_boxes, pred_scores

    def get_predictions(self, cal_dl, test_dl):
        if self._predictions is not None:
            return self._predictions
        # print(
        #    os.path.realpath(f"preds/images_preds_{self._name.lower()}_pretrained.pkl")
        # )
        if not os.path.isfile(
            f"preds/images_preds_{self._name.lower()}_pretrained.pkl"
        ):
            print("Computing predictions")
            images, true_boxes, pred_boxes, scores = self._get_predictions(cal_dl)
            (
                images_test,
                true_boxes_test,
                pred_boxes_test,
                scores_test,
            ) = self._get_predictions(test_dl)
            print("Saving predictions to file")
            with open(
                f"preds/images_preds_{self._name.lower()}_pretrained.pkl", "wb"
            ) as f:  # open a text file
                pickle.dump(
                    (
                        (images, true_boxes, pred_boxes, scores),
                        (images_test, true_boxes_test, pred_boxes_test, scores_test),
                    ),
                    f,
                )  # serialize the list

        else:
            print("Loading predictions from file")
            with open(
                f"preds/images_preds_{self._name.lower()}_pretrained.pkl", "rb"
            ) as f:  # open a text file
                (
                    (images, true_boxes, pred_boxes, scores),
                    (images_test, true_boxes_test, pred_boxes_test, scores_test),
                ) = pickle.load(f)

        pred_boxes = list(
            [
                torch.stack(list(map(lambda x: torch.FloatTensor(x), ls))).cuda()
                for ls in pred_boxes
            ]
        )
        scores = list(
            [torch.FloatTensor(list([x for x in ls])).cuda() for ls in scores]
        )
        pred_boxes_test = list(
            [
                torch.stack(list(map(lambda x: torch.FloatTensor(x), ls))).cuda()
                for ls in pred_boxes_test
            ]
        )
        scores_test = list(
            [torch.FloatTensor(list([x for x in ls])).cuda() for ls in scores_test]
        )

        pred_boxes, scores = nms(pred_boxes, scores, 0.5)
        pred_boxes_test, scores_test = nms(pred_boxes_test, scores_test, 0.5)

        print("Total number of ground truths:", sum([len(x) for x in true_boxes]))
        print("Total number of predictions:", sum([len(x) for x in pred_boxes]))

        self._predictions = (
            Predictions(images, true_boxes, pred_boxes, scores),
            Predictions(images_test, true_boxes_test, pred_boxes_test, scores_test),
        )

        return self._predictions
    
    def load_predictions(self, tqdm_on=True):
        if self._predictions is not None:
            return self._predictions
        if not os.path.isfile(
            f"preds/images_preds_{self._name.lower()}_pretrained.pkl"
        ):
            raise ValueError(f" No such file 'preds/images_preds_{self._name.lower()}_pretrained.pkl'")
        else:
            print("Loading predictions from file")
            with open(
                f"preds/images_preds_{self._name.lower()}_pretrained.pkl", "rb"
            ) as f:  # open a text file
                (
                    (images, true_boxes, pred_boxes, scores),
                    (images_test, true_boxes_test, pred_boxes_test, scores_test),
                ) = pickle.load(f)

        pred_boxes = list(
            [
                torch.stack(list(map(lambda x: torch.FloatTensor(x), ls))).cuda()
                for ls in pred_boxes
            ]
        )
        scores = list(
            [torch.FloatTensor(list([x for x in ls])).cuda() for ls in scores]
        )
        pred_boxes_test = list(
            [
                torch.stack(list(map(lambda x: torch.FloatTensor(x), ls))).cuda()
                for ls in pred_boxes_test
            ]
        )
        scores_test = list(
            [torch.FloatTensor(list([x for x in ls])).cuda() for ls in scores_test]
        )

        pred_boxes, scores = nms(pred_boxes, scores, 0.5)
        pred_boxes_test, scores_test = nms(pred_boxes_test, scores_test, 0.5)

        print("Total number of ground truths:", sum([len(x) for x in true_boxes]))
        print("Total number of predictions:", sum([len(x) for x in pred_boxes]))

        self._predictions = (
            Predictions(images, true_boxes, pred_boxes, scores),
            Predictions(images_test, true_boxes_test, pred_boxes_test, scores_test),
        )

        return self._predictions


class DiffusionDet(AbstractModel):
    def default_cfg(self, other_dataset=None, other_weights=None, other_config=None):
        # load config from file and command-line arguments
        cfg = get_cfg()

        # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
        # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
        # add_panoptic_deeplab_config(cfg)
        add_diffusiondet_config(cfg)
        add_model_ema_configs(cfg)
        cfg.merge_from_file(
            "configs/diffdet.coco.res50.300boxes.yaml"
            if other_config is None
            else other_config
        )
        if other_dataset is not None:
            cfg.DATASETS.TRAIN = other_dataset.get("train", ())
            cfg.DATASETS.TEST = other_dataset.get(
                "test", ()
            )  # no metrics implemented for this dataset
            cfg.SOLVER.MAX_ITER = 10000  # 10000 iterations seems good enough, but you can certainly train longer
            cfg.SOLVER.IMS_PER_BATCH = 3  # This is the real "batch size" commonly known to deep learning people
            # cfg.merge_from_list(args.opts)
            # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3  # 0.5#args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # 0.5#args.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
            0.3  # 0.5#args.confidence_threshold
        )
        cfg.MODEL.DiffusionDet.SAMPLE_STEP = 4
        # cfg.MODEL.DiffusionDet.NUM_PROPOSALS = 1000
        if other_weights is not None:
            cfg.MODEL.WEIGHTS = other_weights
        else:
            cfg.MODEL.WEIGHTS = "models/diffdet_coco_res50_300boxes.pth"
            cfg.freeze()
        return cfg

    def __init__(self):
        super().__init__()
        self.cfg = self.default_cfg()
        self._name = "DiffusionDet"
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, x):
        if len(x) > 1:
            raise ValueError("DiffusionDet only handles a batch size of 1 currently")
        predictions = self.predictor(x[0])

        # processing preds
        instances = predictions["instances"]
        instances = instances[instances.pred_classes == 9]
        # instances = instances[instances.scores>THRESHOLD]
        # no "e" in boxes = the boxes for that sample
        pred_boxs = instances.pred_boxes
        pred_classes = instances.scores
        pred_boxs = list(
            [
                # 24-03 : Pourquoi il y a encore score ici ? cela devait etre enlevé depuis longtemps
                box.detach().cpu().numpy()  # (box.detach().cpu().numpy(), score)
                for box, score in zip(pred_boxs, pred_classes)
            ]
        )

        return pred_boxs, pred_classes


class DETR(AbstractModel):
    def __init__(self):
        super().__init__()
        self._name = "detr"
        self.predictor = torch.hub.load(
            "facebookresearch/detr:main", "detr_resnet50", pretrained=True
        ).cuda()

        # class AddGaussianNoise(object):
        #     def __init__(self, mean=0., std=1.):
        #         self.std = std
        #         self.mean = mean

        #     def __call__(self, tensor):
        #         return tensor + torch.randn(tensor.size()) * self.std + self.mean

        #     def __repr__(self):
        #         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

        # standard PyTorch mean-std input image normalization
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.transform_train = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(800),
                # T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ToTensor(),
                # AddGaussianNoise(0.1),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def box_cxcywh_to_xyxy_np(self, x):
        x_c = x[:, 0]
        y_c = x[:, 1]
        w = x[:, 2]
        h = x[:, 3]
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return np.asarray(b).transpose()

    def box_xyxy_to_cxcywh(self, x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(
            b.device
        )
        return b

    def box_switch_to_pixels(self, image, box):
        he, wi, chan = image.shape
        converted = self.box_cxcywh_to_xyxy_np(box) * np.array([wi, he, wi, he])

        return converted

    def predict(self, x):
        if len(x) > 1:
            raise ValueError("DETR only handles a batch size of 1 currently")

        self.predictor.eval()
        xi = x[0]
        xi_or = xi
        # xi = np.moveaxis(xi, 2, 0)
        xi = self.transform(xi).unsqueeze(0).cuda()
        # xi = torch.FloatTensor(xi[None]).cuda()
        # print(xi.shape)
        predictions = self.predictor(xi)
        # print(predictions)

        # processing preds
        # instances = predictions
        # instances = instances[instances.pred_classes==9]
        # instances = instances[instances.scores>THRESHOLD]
        # no "e" in boxes = the boxes for that sample
        # pred_boxs = instances.pred_boxes
        # pred_classes = instances.scores

        predictions["boxes"] = (
            predictions["pred_boxes"].squeeze(0).detach().cpu().numpy()
        )

        temp = torch.nn.functional.softmax(predictions["pred_logits"].squeeze(0), dim=1)
        predictions["confidence"] = temp.detach().cpu().numpy()

        predictions["class"] = np.argmax(predictions["confidence"], axis=1)
        # print( predictions_values['pred_logits'])
        valid_signals = np.nonzero(predictions["class"] == 9 + 1)[0]
        # print(valid_signals.shape)
        predictions["confidence"] = np.max(predictions["confidence"], axis=1)[
            valid_signals
        ]
        predictions["boxes"] = predictions["boxes"][valid_signals, :]
        predictions["boxes"] = self.box_switch_to_pixels(xi_or, predictions["boxes"])
        predictions["class"] = predictions["class"][valid_signals]

        # pred_classes = predictions['pred_logits'].softmax(-1)[0, :, :-1]
        # keep = torch.argmax(pred_classes, 1)==9
        pred_classes = predictions["confidence"]  # pred_classes[keep][:,9]
        pred_boxs = predictions[
            "boxes"
        ]  # rescale_bboxes(predictions['pred_boxes'][0, keep], (xi.shape[3], xi.shape[2]))

        return pred_boxs, pred_classes


class YOLOv5(AbstractModel):
    def __init__(self):
        super().__init__()
        self._name = "Yolov5CIM"
        model = torch.load("./best.pt")["model"].cuda()  # .to(torch.float32).cuda()
        model.model[11].recompute_scale_factor = None
        model.model[15].recompute_scale_factor = None
        self.predictor = model

    def predict(self, x):
        if len(x) > 1:
            raise ValueError("DiffusionDet only handles a batch size of 1 currently")

        with torch.cuda.amp.autocast():
            x = torch.randn((1, 3, 256, 256)).cuda()
            y, _ = self.predictor(x)
            pred_boxs = y[0, :, :4].detach().cpu().numpy()
            pred_scores = y[0, :, 4].detach().cpu().numpy()

        return pred_boxs, pred_scores

    def letterbox(
        self,
        im,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=True,
        scaleFill=False,
        scaleup=True,
        stride=32,
    ):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[2:]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return im, ratio, (dw, dh)
