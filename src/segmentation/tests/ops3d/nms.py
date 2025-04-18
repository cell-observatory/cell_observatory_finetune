import torch
import ops3d._C as _C

def main():
    iou_thresh = 0.1
    box0 = torch.tensor([0,0,0, 100,100,100], device="cuda")  # high score
    boxes = box0.unsqueeze(0).repeat(1000, 1)  # (1000, 7)
    
    score_0 = torch.tensor([0.9], device="cuda")  # scores for each box
    scores = torch.cat([score_0, torch.zeros(999, device="cuda")], dim=0)  # (1000,)

    keep = _C.nms_3d(boxes, scores, iou_thresh)

    print("kept indices:", keep.cpu().tolist())      # → [0]  (only the higher‑score copy)
    print("num kept:", len(keep))    

if __name__ == "__main__":
    main()
