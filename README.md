# Image Captioning with Structured Object Features

This repository contains a notebook-driven image captioning pipeline that combines object detection with structured scene understanding. The project compares multiple detectors, prepares object-aware feature tensors, and evaluates caption generation models on Flickr8K + Flickr30K data.

## Project goal

The core idea is to improve image captioning by incorporating structured object information rather than relying only on raw image features. The workflow includes:

- object detection with YOLOv5s, YOLO26s, and RT-DETR-X
- feature preparation for object-level representations
- structured object tensor generation for captioning models
- training and evaluation of an object-aware transformer caption generator
- benchmark comparison across detection models

## Key results from saved outputs

The repository includes evaluation artifacts generated during experimentation:

- Detector comparison ranking: RT-DETR-X > YOLO26s > YOLOv5s
- RT-DETR-X detector coverage on 39,874 images: 99.985%
- RT-DETR-X zero-detection images: 6
- RT-DETR-X average detections per image: 12.46
- Final structured captioning model using RT-DETR-X:
  - BLEU-1: 0.753674
  - BLEU-2: 0.395545
  - BLEU-3: 0.258459
  - BLEU-4: 0.173117
  - METEOR: 0.388903
  - ROUGE-L: 0.457393
  - CIDEr: 0.360418
- YOLO26s Object-Aware Transformer evaluation output:
  - BLEU-4: 0.1724
  - ROUGE-L: 0.4072
  - CIDEr: 1.0324

These results are stored in:

- `Outputs/Model_Comparison/model_comparison_report.json`
- `Outputs/Structured_Captioning/Evaluation/final_evaluation_results.json`
- `Outputs/YOLO26_Object_Aware_Transformer/Evaluation/yolo26_transformer_final_metrics.json`

## Repository structure

```text
.
├── Features/
│   ├── Structured_Object/
│   │   ├── caption_sequences.npz
│   │   ├── caption_sequences_train_only.npz
│   │   ├── caption_sequences_corrected.npz
│   │   ├── rtdetr_structured_object_features.npz
│   │   └── ...
│   ├── YOLO26_Object_Aware_Transformer/
│   │   └── yolo26_structured_object_features.npz
│   └── YOLOv5/
│       └── structured_objects_cache.pt
├── Notebooks/
│   ├── 01_Unzip_Datasets.ipynb
│   ├── 02_Merge_Dataset.ipynb
│   ├── 03_Caption_Preprocessing.ipynb
│   ├── 04_Image_Preparation.ipynb
│   ├── 05_YOLOv5_Detection.ipynb
│   ├── 06_YOLOv5_Feature_Preparation.ipynb
│   ├── 07_MODEL_2_YOLO26s_FEATURE_EXTRACTION.ipynb
│   ├── 08_RT_DETR_X_OBJECT_DETECTION.ipynb
│   ├── 09_OBJECT_DETECTION_MODEL_COMPARISON.ipynb
│   ├── 10_STRUCTURED_OBJECT_CAPTION_PREPARATION.ipynb
│   ├── 11_OBJECT_AWARE_TRANSFORMER_CAPTION_GENERATOR.ipynb
│   ├── 12_CLEAN_OBJECT_AWARE_TRANSFORMER_RESUME_EVALUATION.ipynb
│   └── 13_YOLO26_Object_Aware_Transformer_DriveOnly.ipynb
├── Outputs/
│   ├── Model_Comparison/
│   ├── RTDETR_Detection/
│   ├── Structured_Captioning/
│   ├── YOLO26_Detection/
│   ├── YOLO26_Object_Aware_Transformer/
│   └── YOLOv5_Detection/
├── README.md
└── .git/
```

## Dataset and preprocessing

The project is based on the Flickr8K and Flickr30K image-caption datasets, combined into a single processed collection. The saved configuration indicates:

- total images: 39,874
- total captions: 199,369
- train / validation / test split: 80 / 10 / 10
- random seed: 42
- max caption length: 32
- top object detections per image retained: 10
- object confidence threshold: 0.3
- object vocabulary size: 82
- caption vocabulary size: 12,174

These settings are defined in:

- `Outputs/Structured_Captioning/structured_captioning_config.json`

## Detection model comparison

The project evaluates three detectors on the same image corpus:

1. YOLOv5s
2. YOLO26s
3. RT-DETR-X

The comparison ranks them by coverage, number of detections, average detections per image, and class diversity. The saved report shows that RT-DETR-X has the strongest object coverage and the largest detection richness, while YOLO26s remains a close second.

## Captioning pipeline

The captioning pipeline is built in notebook stages:

1. `01_Unzip_Datasets.ipynb` — prepare dataset files
2. `02_Merge_Dataset.ipynb` — combine Flickr8K + Flickr30K
3. `03_Caption_Preprocessing.ipynb` — normalize captions and tokenization setup
4. `04_Image_Preparation.ipynb` — prepare standardized image inputs
5. `05_YOLOv5_Detection.ipynb` — run YOLOv5 detection
6. `06_YOLOv5_Feature_Preparation.ipynb` — convert detections into model-ready features
7. `07_MODEL_2_YOLO26s_FEATURE_EXTRACTION.ipynb` — extract YOLO26s object features
8. `08_RT_DETR_X_OBJECT_DETECTION.ipynb` — run RT-DETR-X detection
9. `09_OBJECT_DETECTION_MODEL_COMPARISON.ipynb` — compare detector outputs
10. `10_STRUCTURED_OBJECT_CAPTION_PREPARATION.ipynb` — create structured object-caption data
11. `11_OBJECT_AWARE_TRANSFORMER_CAPTION_GENERATOR.ipynb` — train/evaluate object-aware transformer
12. `12_CLEAN_OBJECT_AWARE_TRANSFORMER_RESUME_EVALUATION.ipynb` — evaluation cleanup/resume workflow
13. `13_YOLO26_Object_Aware_Transformer_DriveOnly.ipynb` — YOLO26-based captioning variant

## Outputs and artifacts

Important generated artifacts are stored under `Outputs/`:

- `Outputs/Model_Comparison/` — detection rankings and benchmark summaries
- `Outputs/RTDETR_Detection/` — RT-DETR-X detection CSVs and checkpoints
- `Outputs/YOLO26_Detection/` — YOLO26s detection outputs
- `Outputs/YOLOv5_Detection/` — YOLOv5 detection outputs
- `Outputs/Structured_Captioning/` — caption vocabularies, sequence data, checkpoints, and evaluation summaries
- `Outputs/YOLO26_Object_Aware_Transformer/` — YOLO26s captioning evaluation outputs

## Requirements and environment

This project is notebook-based and uses a research stack rather than a single packaging entry point. Typical dependencies include:

- Python 3.10+
- PyTorch
- Ultralytics / YOLOv5-compatible environment
- NumPy, pandas
- Matplotlib
- NLTK / evaluation libraries
- OpenCV / PIL for image processing

For reproducing the work, open the notebooks in order and keep the repository paths consistent, especially the `Features/` and `Outputs/` folders.

## How to use this project

1. Clone the repository.
2. Place the dataset files in the expected working directory structure.
3. Run the notebooks in order from `01_` through `13_`.
4. Check generated artifacts under `Features/` and `Outputs/`.
5. Use model comparison and evaluation JSON files to compare detector and captioning performance.

## Notes

- The project is exploratory and research-oriented rather than a polished library package.
- The notebook sequence is the main source of truth for the full pipeline.
- Final evaluation files in `Outputs/` should be treated as the canonical references for performance numbers.
