import os
import argparse
import torch
import cv2
import pandas as pd
import numpy as np
from skimage.measure import regionprops, label
from pycocotools.coco import COCO
from src.model import UNet
from src.metrics import calculate_iou, calculate_dice, calculate_precision_recall_f1

def get_gt_mask_from_coco(coco, img_id, h, w):
    """
    Generates a 2-channel mask based on the coco JSON:
    ID 1 -> IM (Inner Membrane) mapped to Channel 1
    ID 2 -> OM (Outer Membrane) mapped to Channel 0
    """
    mask = np.zeros((2, h, w), dtype=np.uint8)
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    for ann in anns:
        m = coco.annToMask(ann)
        if ann['category_id'] == 2:   # OM
            mask[0] = np.maximum(mask[0], m) 
        elif ann['category_id'] == 1: # IM
            mask[1] = np.maximum(mask[1], m)
    return mask

def process_image(image_path, model, device, nm_per_pix, gt_mask=None):
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        return None
        
    h_orig, w_orig = raw_img.shape
    # Resize for U-Net input
    img_input = cv2.resize(raw_img, (640, 640))
    img_tensor = torch.from_numpy(img_input).float().unsqueeze(0).unsqueeze(0).to(device)
    img_tensor = (img_tensor / 255.0 - 0.5) / 0.5

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.sigmoid(output)
        # Match original resolution for accurate area math
        mask_upsampled = torch.nn.functional.interpolate(probs, size=(h_orig, w_orig), mode='bilinear')
        pred_mask = (mask_upsampled > 0.5).cpu().numpy()[0]

    # --- Morphometrics ---
    om_mask, im_mask = pred_mask[0].astype(np.uint8), pred_mask[1].astype(np.uint8)
    om_props = regionprops(label(om_mask))
    
    data = {"Filename": os.path.basename(image_path)}
    
    if om_props:
        om_region = max(om_props, key=lambda r: r.area)
        im_props = regionprops(label(im_mask))
        im_area_px = max(im_props, key=lambda r: r.area).area if im_props else 0
        
        data.update({
            "OM_Area_nm2": om_region.area * (nm_per_pix**2),
            "IM_Area_nm2": im_area_px * (nm_per_pix**2),
            "Periplasm_Area_nm2": (om_region.area - im_area_px) * (nm_per_pix**2),
            "Eccentricity": om_region.eccentricity,
            "Perimeter_nm": om_region.perimeter * nm_per_pix
        })

    # --- Metrics (If ground truth from COCO is provided) ---
    if gt_mask is not None:
        iou = calculate_iou(pred_mask, gt_mask)
        dice = calculate_dice(pred_mask, gt_mask)
        prec, rec, f1 = calculate_precision_recall_f1(pred_mask, gt_mask)
        data.update({"Dice": dice, "IoU": iou, "F1": f1})

    return data, om_mask, im_mask, raw_img

def main():
    parser = argparse.ArgumentParser(description="Bacterial Morphometry Toolkit")
    parser.add_argument("--input", required=True, help="Path to folder containing images")
    parser.add_argument("--weights", default="models/best_unet.pt", help="Path to model weights")
    parser.add_argument("--coco_json", help="Path to _annotations.coco.json for benchmarking")
    parser.add_argument("--nm_pix", type=float, default=0.898, help="Calibration: nm per pixel")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Load COCO database if json path is provided
    coco = COCO(args.coco_json) if args.coco_json else None
    
    os.makedirs("results", exist_ok=True)
    all_results = []

    files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in files:
        f_path = os.path.join(args.input, filename)
        
        # Get Ground Truth mask if in benchmarking mode
        gt_mask = None
        if coco:
            # Search for the image in the COCO JSON by filename
            img_ids = [i for i, img in coco.imgs.items() if img['file_name'] == filename]
            if img_ids:
                img_info = coco.loadImgs(img_ids[0])[0]
                gt_mask = get_gt_mask_from_coco(coco, img_ids[0], img_info['height'], img_info['width'])
        
        result_data, om_m, im_m, raw = process_image(f_path, model, device, args.nm_pix, gt_mask)
        
        if result_data:
            all_results.append(result_data)
            # Save Visualization Trace
            overlay = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            # Red contour for OM, Blue for IM
            cv2.drawContours(overlay, cv2.findContours(om_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0,0,255), 2)
            cv2.drawContours(overlay, cv2.findContours(im_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255,0,0), 2)
            cv2.imwrite(f"results/trace_{filename}", overlay)

    # Save all biological and accuracy data to CSV
    pd.DataFrame(all_results).to_csv("results/morphometry_report.csv", index=False)
    print(f"Processed {len(all_results)} images. Results saved in 'results/' folder.")

if __name__ == "__main__":
    main()