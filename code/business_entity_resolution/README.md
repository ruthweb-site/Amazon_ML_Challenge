# Business Entity Resolution Pipeline (Amazon ML Challenge 2026)

This repository contains the end-to-end machine learning pipeline for resolving business identities across multiple noisy data sources (Source 1, Source 2, and Source 3).

## Directory Structure
```
code/
└── business_entity_resolution/
    ├── src/
    │   ├── preprocessor.py   # Cleans, strips legal suffixes, normalizes addresses
    │   ├── blocking.py       # Inverted index candidate generation (Country + Token/N-gram keys)
    │   ├── matching.py       # RapidFuzz / String similarity scoring
    │   ├── evaluate.py       # Macro-averaged F_0.5 metric computation
    │   └── pipeline.py       # End-to-end execution script
    ├── requirements.txt      # Pinned dependencies
    └── README.md             # Pipeline documentation
```

## How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline on Test Data
To generate `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
```bash
python src/pipeline.py \
    --s1 ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv \
    --s2 ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source2.tsv \
    --s3 ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source3.tsv \
    --output-dir ../../output \
    --threshold 0.75
```

### 3. Validate Before Submission
Always run the official validator before uploading:
```bash
python ../../6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test
```

## Running on AWS SageMaker

Inside your SageMaker JupyterLab terminal:
```bash
# 1. Unzip the dataset
unzip 6ab10eb3b23ba_student_resource.zip

# 2. Run pipeline over full test dataset
python code/business_entity_resolution/src/pipeline.py \
    --s1 student_resource/dataset/test/test_source1.tsv \
    --s2 student_resource/dataset/test/test_source2.tsv \
    --s3 student_resource/dataset/test/test_source3.tsv \
    --output-dir output \
    --threshold 0.75
```
