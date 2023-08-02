import json
from collections import defaultdict

from tools.JSON2YOLO.utils import *
from tools.JSON2YOLO.general_json2yolo import merge_multi_segment


def convert(json_filepath, output_dir, use_segments=False, cls91to80=False):
    coco80 = coco91_to_coco80_class()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_filepath) as f:
        data = json.load(f)

    # Create image dict
    images = {'%g' % x['id']: x for x in data['images']}
    # Create image-annotations dict
    imgToAnns = defaultdict(list)
    for ann in data['annotations']:
        imgToAnns[ann['image_id']].append(ann)

    # Write labels file
    for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {json_filepath}'):
        img = images['%g' % img_id]
        h, w, f = img['height'], img['width'], img['file_name']

        # Check that there is at least 1 segment if use_segments is passed
        contains_seg = False
        if use_segments:
            for ann in anns:
                if ann['segmentation']:
                    contains_seg = True
                    break

        bboxes = []
        segments = []
        for ann in anns:
            if ann['iscrowd']:
                continue
            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(ann['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue

            cls = coco80[ann['category_id'] - 1] if cls91to80 else ann['category_id']  # class
            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)
            # Segments
            if use_segments and contains_seg:
                # if no segmentation for a specific annotation
                if not ann['segmentation']:
                    s = np.array(ann['bbox'], dtype=np.float64)
                    s[2] += s[0]
                    s[3] += s[1]
                    s[[0, 2]] /= w  # normalize x
                    s[[1, 3]] /= h  # normalize y

                    s = s.tolist()

                elif len(ann['segmentation']) > 1:
                    s = merge_multi_segment(ann['segmentation'])
                    s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                else:
                    s = [j for i in ann['segmentation'] for j in i]  # all segments concatenated
                    s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                s = [cls] + s
                if s not in segments:
                    segments.append(s)

        # Write
        with open((output_dir / f).with_suffix('.txt'), 'w') as file:
            for i in range(len(bboxes)):
                line = *(segments[i] if (use_segments and contains_seg) else bboxes[i]),  # cls, box or segments
                file.write(('%g ' * len(line)).rstrip() % line + '\n')