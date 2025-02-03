import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd

# File paths
image_path = r".\data\images\BloodImage_00000.jpg"
xml_path = r".\data\annotations\BloodImage_00000.xml"
csv_path = r".\data\labels.csv"

# Load image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib


# Parse XML annotations
def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = []
    for obj in root.findall(".//object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        objects.append((name, xmin, ymin, xmax, ymax))
    return objects


annotations = parse_annotations(xml_path)

# Draw bounding boxes on the image
for obj in annotations:
    name, xmin, ymin, xmax, ymax = obj
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv2.putText(
        image, name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
    )

# Load CSV labels
df = pd.read_csv(csv_path)
image_label = df[
    df["Image"].apply(lambda x: f"BloodImage_{x:05d}.jpg") == "BloodImage_00000.jpg"
]["Category"].values[0]

# Display image with annotations
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.axis("off")
plt.title(f"Label: {image_label}")
plt.show()
