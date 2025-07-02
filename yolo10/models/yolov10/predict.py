from yolo10.models.yolo.detect import DetectionPredictor
import torch
from yolo10.utils import ops
from yolo10.engine.results import Results


class YOLOv10DetectionPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        if preds.shape[-1] == 6:
            pass
        else:
            preds = preds.transpose(-1, -2)
            bboxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, preds.shape[-1]-4)
            bboxes = ops.xywh2xyxy(bboxes)
            preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

        mask = preds[..., 4] > self.args.conf
        if self.args.classes is not None:
            mask = mask & (preds[..., 5:6] == torch.tensor(self.args.classes, device=preds.device).unsqueeze(0)).any(2)
        
        preds = [p[mask[idx]] for idx, p in enumerate(preds)]



        nms_preds = []
        for p in preds:
            if p.shape[0] > 0:
                p = _nms_handwritten(p, iou_threshold= 0.1)
            nms_preds.append(p)
        preds = nms_preds
        #getattr(self.args, "iou", 0.1


        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results



def _nms_handwritten(boxes_scores_labels, iou_threshold=0.1):
    if boxes_scores_labels.size(0) == 0:
        return boxes_scores_labels

    boxes = boxes_scores_labels[:, :4]
    scores = boxes_scores_labels[:, 4]
    labels = boxes_scores_labels[:, 5]

    # 按 score 降序排序
    indices = scores.argsort(descending=True)
    keep = []

    while indices.numel() > 0:
        # 取出当前最高分框的 index
        current = indices[0]
        keep.append(current.item())

        if indices.numel() == 1:
            break

        # 当前框与其他框计算 IOU
        current_box = boxes[current].unsqueeze(0)  # [1, 4]
        other_boxes = boxes[indices[1:]]           # [N-1, 4]

        # 计算IOU
        x1 = torch.max(current_box[:, 0], other_boxes[:, 0])
        y1 = torch.max(current_box[:, 1], other_boxes[:, 1])
        x2 = torch.min(current_box[:, 2], other_boxes[:, 2])
        y2 = torch.min(current_box[:, 3], other_boxes[:, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1 = (current_box[:, 2] - current_box[:, 0]) * (current_box[:, 3] - current_box[:, 1])
        area2 = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        iou = inter / (area1 + area2 - inter + 1e-6)

        # 筛掉与当前框 IOU 过高的索引
        below_thresh = iou <= iou_threshold
        indices = indices[1:][below_thresh]

    return boxes_scores_labels[keep]