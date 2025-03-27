# JSON to YOLO Segmentation Converter

This repository contains a Python script (`convert_json_to_yolo.py`) that converts JSON-formatted segmentation labels into YOLO segmentation format. It supports both post-disaster JSON files (with damage classes) and pre-disaster JSON files (without damage classes).

## Overview

The script reads JSON files containing polygon annotations for buildings in two different formats:
- **Post-disaster JSON:** Each building object includes a `"subtype"` field. A class mapping is applied:
  - `"no-damage": 0`
  - `"minor-damage": 1`
  - `"major-damage": 2`
  - `"destroyed": 3`
- **Pre-disaster JSON:** Since there is no damage information, a default class id (0) is assigned to all objects.

The output is a set of text files containing YOLO-formatted labels with normalized coordinates. The YOLO segmentation format for each object is:
```
<class_id> x1 y1 x2 y2 ... xn yn
```

## Dataset Structure

The script expects the input dataset to be organized as follows:

```graphql
<data_root>/segmentation_dataset/
    train/
        labels/
            image_id_*.json
    val/
        labels/
            image_id_*.json
    test/
        labels/
            image_id_*.json
```

The output YOLO labels will be created in a directory structure as follows:
Post disaster labels:
```graphql
<dst_root_post>/
    train/
        labels/
            image_id_*.txt
    val/
        labels/
            image_id_*.txt
    test/
        labels/
            image_id_*.txt
```
Pre disaster labels:
```graphql
<dst_root_pre>/
    train/
        labels/
            image_id_*.txt
    val/
        labels/
            image_id_*.txt
    test/
        labels/
            image_id_*.txt
```

## Prerequisites

- Python 3.x
- No additional packages are required as the script uses Python’s standard libraries (`os`, `json`, and `argparse`).


## Usage

The script uses command-line arguments to specify the input and output directories. You can run the script from your project root directory as follows:

```bash
python scripts/convert_json_to_yolo.py --src_root data/segmentation_dataset --dst_root yolo
```