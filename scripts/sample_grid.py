import os
import random
from PIL import Image, ImageDraw, ImageFont

DATASET_ROOT = 'terrain_dataset'
IMAGE_FOLDERS = [os.path.join(DATASET_ROOT, p) for p in ['train/images', 'valid/images', 'test/images']]
LABEL_FOLDERS = [os.path.join(DATASET_ROOT, p) for p in ['train/labels', 'valid/labels', 'test/labels']]
CLASSES_FILE = 'classes.txt'
OUT_DIR = 'reports'
os.makedirs(OUT_DIR, exist_ok=True)

COLS = 4
IMG_W = 256
IMG_H = 192
PADDING = 8
LABEL_COL_W = 320
FONT_SIZE = 56


def load_large_font(size):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()

def load_classes():
    if not os.path.exists(CLASSES_FILE):
        return []
    with open(CLASSES_FILE, 'r') as f:
        return [l.strip() for l in f.readlines() if l.strip()]

def find_image_for_label(label_file):
    # label_file: absolute path to .txt under labels folder
    stem = os.path.splitext(os.path.basename(label_file))[0]
    # search for images with same stem in image folders
    for folder in IMAGE_FOLDERS:
        if not os.path.isdir(folder):
            continue
        for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tif'):
            p = os.path.join(folder, stem + ext)
            if os.path.exists(p):
                return p
    return None

def collect_examples_per_class(classes, max_per_class=COLS):
    examples = {c: [] for c in classes}
    # scan label folders
    for lab_folder in LABEL_FOLDERS:
        if not os.path.isdir(lab_folder):
            continue
        for fname in os.listdir(lab_folder):
            if not fname.endswith('.txt'):
                continue
            lab_path = os.path.join(lab_folder, fname)
            try:
                with open(lab_path, 'r') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
            except Exception:
                continue
            if not lines:
                continue
            # get all class ids present
            ids = set()
            for L in lines:
                parts = L.split()
                if len(parts) == 0:
                    continue
                try:
                    cid = int(float(parts[0]))
                except Exception:
                    continue
                ids.add(cid)

            img_path = find_image_for_label(lab_path)
            if img_path is None:
                continue
            for cid in ids:
                if cid < 0 or cid >= len(classes):
                    continue
                cname = classes[cid]
                if len(examples[cname]) < max_per_class:
                    # store tuple (image_path, label_path, class_id) so we can draw boxes later
                    examples[cname].append((img_path, lab_path, cid))
    # for classes with fewer examples, try duplicating or leaving blanks
    return examples

def make_grid(examples, out_path):
    classes = list(examples.keys())
    rows = len(classes)
    cols = COLS
    canvas_w = cols * IMG_W + (cols + 1) * PADDING + LABEL_COL_W
    canvas_h = rows * IMG_H + (rows + 1) * PADDING

    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255,255,255))
    draw = ImageDraw.Draw(canvas)
    font = load_large_font(FONT_SIZE)

    for r, cname in enumerate(classes):
        # draw class label on left column area
        y0 = PADDING + r * (IMG_H + PADDING)
        label_box = (PADDING, y0, PADDING + LABEL_COL_W, y0 + IMG_H)
        # center text vertically
        draw.rectangle(label_box, fill=(245,245,245))
        try:
            bbox = draw.textbbox((0,0), cname, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            w, h = font.getsize(cname)
        tx = PADDING + (LABEL_COL_W - w) // 2
        ty = y0 + (IMG_H - h)//2
        draw.text((tx, ty), cname, fill=(0,0,0), font=font)

        imgs = examples.get(cname, [])
        for c in range(cols):
            x = PADDING + LABEL_COL_W + PADDING + c * (IMG_W + PADDING)
            y = y0
            if c < len(imgs):
                img_path, lab_path, cid = imgs[c]
                try:
                    im = Image.open(img_path).convert('RGB')
                    orig_w, orig_h = im.size
                    # draw boxes on a copy then resize to grid size
                    im_draw = im.copy()
                    idraw = ImageDraw.Draw(im_draw)
                    # parse label file and draw boxes for this class id
                    try:
                        with open(lab_path, 'r') as lf:
                            for line in lf:
                                parts = line.strip().split()
                                if len(parts) < 5:
                                    continue
                                try:
                                    cls_id = int(float(parts[0]))
                                except Exception:
                                    continue
                                if cls_id != cid:
                                    continue
                                xc = float(parts[1]) * orig_w
                                yc = float(parts[2]) * orig_h
                                bw = float(parts[3]) * orig_w
                                bh = float(parts[4]) * orig_h
                                x0 = xc - bw/2
                                y0b = yc - bh/2
                                x1 = xc + bw/2
                                y1 = yc + bh/2
                                # draw rectangle
                                idraw.rectangle([x0, y0b, x1, y1], outline=(255,0,0), width=4)
                                # label text
                                try:
                                    idraw.text((x0+4, y0b+2), classes[cid], fill=(255,255,255))
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    im = im_draw.resize((IMG_W, IMG_H))
                    canvas.paste(im, (x, y))
                except Exception:
                    # draw placeholder
                    draw.rectangle([x, y, x+IMG_W, y+IMG_H], outline=(0,0,0))
                    draw.line([x, y, x+IMG_W, y+IMG_H], fill=(0,0,0))
            else:
                # blank placeholder
                draw.rectangle([x, y, x+IMG_W, y+IMG_H], outline=(200,200,200))

    canvas.save(out_path)

def main():
    classes = load_classes()
    if not classes:
        print('No classes found in', CLASSES_FILE)
        return
    examples = collect_examples_per_class(classes, max_per_class=COLS)
    out = os.path.join(OUT_DIR, 'dataset_samples.png')
    make_grid(examples, out)
    print('Saved', out)

if __name__ == '__main__':
    main()
