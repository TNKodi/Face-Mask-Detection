# import torch

# if torch.cuda.is_available():
#     print("CUDA is supported!")
#     print("Device Count:", torch.cuda.device_count())
#     print("Device Name:", torch.cuda.get_device_name(0))
#     print("CUDA Version:", torch.version.cuda)
# else:
#     print("CUDA is NOT supported.")
import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# — adjust these if your structure is different —
IMG_DIR = os.path.join('dataset', 'images')
ANN_DIR = os.path.join('dataset', 'annotations')

# 1) List all files
images   = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))])
annots   = sorted([f for f in os.listdir(ANN_DIR) if f.lower().endswith('.xml')])
 
print(f"Found {len(images)} images in `{IMG_DIR}`")
print(f"Found {len(annots)} annotations in `{ANN_DIR}`\n")

# 2) Check for unmatched files
missing_xml = [img for img in images if img.rsplit('.',1)[0] + '.xml' not in annots]
missing_img = [ann for ann in annots if ann.rsplit('.',1)[0] + '.png' not in images]

if missing_xml:
    print("⚠️ Images missing XML:", missing_xml[:5], "…")
else:
    print("✅ Every image has a matching XML")

if missing_img:
    print("⚠️ XMLs missing images:", missing_img[:5], "…")
else:
    print("✅ Every XML has a matching image")

# 3) Visualize a few samples
def parse_boxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        bb = obj.find('bndbox')
        xmin = int(bb.find('xmin').text)
        ymin = int(bb.find('ymin').text)
        xmax = int(bb.find('xmax').text)
        ymax = int(bb.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes

for sample in images[:3]:  # first 3 images
    img_path = os.path.join(IMG_DIR, sample)
    xml_path = os.path.join(ANN_DIR, sample.rsplit('.',1)[0] + '.xml')
    
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    boxes = parse_boxes(xml_path)
    
    # draw boxes
    for (xmin, ymin, xmax, ymax) in boxes:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
    
    # show
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.title(sample)
    plt.axis('off')

plt.show()
