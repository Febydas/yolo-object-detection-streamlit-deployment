import os
from PIL import Image

def parse_polygon(polygon_str):
    """Parse a string of polygon coordinates into a list of (x, y) tuples."""
    coords = list(map(float, polygon_str.split()))
    return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

def get_bbox_from_polygon(polygon_points):
    """Calculate bounding box from polygon points."""
    x_coords, y_coords = zip(*polygon_points)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return x_min, y_min, x_max, y_max

def convert_to_yolo_format(x_min, y_min, x_max, y_max, img_width, img_height):
    """Convert bounding box coordinates to YOLO format."""
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

def process_annotation_file(annotation_file, img_width, img_height):
    """Process a single annotation file and convert it to YOLO format."""
    bboxes = []
    try:
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                polygon_points = parse_polygon(line.strip())
                x_min, y_min, x_max, y_max = get_bbox_from_polygon(polygon_points)
                yolo_bbox = convert_to_yolo_format(x_min, y_min, x_max, y_max, img_width, img_height)
                bboxes.append(yolo_bbox)
    except Exception as e:
        print(f"Error processing {annotation_file}: {e}")
    return bboxes

def process_split(split_dir, image_ext='.jpg'):
    """Process a dataset split directory to convert annotations to YOLO format."""
    if not os.path.exists(split_dir):
        print(f"Directory {split_dir} does not exist.")
        return

    for file_name in os.listdir(split_dir):
        if file_name.endswith('.txt'):
            print(f"Processing {file_name}...")
            annotation_file = os.path.join(split_dir, file_name)
            img_name = file_name.replace('.txt', image_ext)
            img_path = os.path.join(split_dir, img_name)

            if not os.path.isfile(img_path):
                print(f"Warning: Image {img_path} not found. Skipping.")
                continue

            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size

                bboxes = process_annotation_file(annotation_file, img_width, img_height)

                label_file = os.path.join(split_dir, file_name)
                with open(label_file, 'w') as f:
                    for bbox in bboxes:
                        f.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

                print(f"Finished processing {file_name}.")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

# Define the directories and process them
for split in ['train', 'test', 'valid']:
    split_directory = f'C:/WORLD/yolo-env/prj1/crack-seg/{split}'
    process_split(split_directory, image_ext='.jpg')

