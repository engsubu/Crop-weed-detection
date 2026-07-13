import cv2
import os

def visualize_labels(img_path, label_path, class_names):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            cls, x, y, bw, bh = map(float, line.split())
            # Convert from YOLO format to pixels
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)
            
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, class_names[int(cls)], (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imshow("Labeled Image", img)
    cv2.waitKey(0)

# Example usage: visualize_labels('data/images/train/img1.jpg', 'data/labels/train/img1.txt', ['crop', 'weed'])
