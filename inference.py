import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

from dataset import (
    DATA_ROOT,
    PATCH_SIZE,
    build_data_list,
    get_val_transforms,
    split_holdout_data_list,
)
from model import build_model

CHECKPOINT = "brats_unet_mps.pth"
VAL_FRACTION = 0.2
HOLDOUT_SEED = 42
NUM_WORKERS = 0
PROB_THRESHOLD = 0.5
SW_BATCH_SIZE = 1
SW_OVERLAP = 0.25


def dice_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.cpu()
    gt = gt.cpu()
    pred_sum = pred.sum().item()
    gt_sum = gt.sum().item()
    if pred_sum + gt_sum == 0:
        return 1.0
    intersection = (pred & gt).sum().item()
    return (2.0 * intersection) / (pred_sum + gt_sum)


def regions_from_prediction(pred_binary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # pred_binary: [3, H, W, D] for classes 1 (NCR), 2 (ED), 3 (ET)
    ncr = pred_binary[0].bool()
    ed = pred_binary[1].bool()
    et = pred_binary[2].bool()
    wt = ncr | ed | et
    tc = ncr | et
    return wt, tc, et


def regions_from_label(label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # label: [1, H, W, D] with values 0..3
    label_map = label.squeeze(0)
    wt = (label_map == 1) | (label_map == 2) | (label_map == 3)
    tc = (label_map == 1) | (label_map == 3)
    et = label_map == 3
    return wt, tc, et


def evaluate() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data_list = build_data_list(DATA_ROOT)
    _, val_list = split_holdout_data_list(data_list, val_fraction=VAL_FRACTION, seed=HOLDOUT_SEED)
    loader = DataLoader(
        Dataset(data=val_list, transform=get_val_transforms()),
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = build_model().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    wt_scores: list[float] = []
    tc_scores: list[float] = []
    et_scores: list[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]

            logits = sliding_window_inference(
                inputs=images,
                roi_size=PATCH_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=model,
                overlap=SW_OVERLAP,
            )
            probs = torch.sigmoid(logits)
            pred_binary = probs > PROB_THRESHOLD

            pred_wt, pred_tc, pred_et = regions_from_prediction(pred_binary[0])
            gt_wt, gt_tc, gt_et = regions_from_label(labels[0])

            wt_scores.append(dice_score(pred_wt, gt_wt))
            tc_scores.append(dice_score(pred_tc, gt_tc))
            et_scores.append(dice_score(pred_et, gt_et))

    mean_wt = sum(wt_scores) / len(wt_scores)
    mean_tc = sum(tc_scores) / len(tc_scores)
    mean_et = sum(et_scores) / len(et_scores)

    print(f"Mean Dice — WT: {mean_wt:.4f} | TC: {mean_tc:.4f} | ET: {mean_et:.4f}")


if __name__ == "__main__":
    evaluate()
