# GreenSpace_CNN

## Overview
Train a CNN to reproduce greenspace ratings at scale using human judgments from photos/satellite images.

## Survey Design
Human raters evaluate images across standardized criteria:

**Core Rating:**
- `structured_unstructured_rating`: 1-5 scale (1=very structured, 5=very unstructured)

**Binary Presence Items:** (Yes/No/Can't answer)
- Sports fields, multipurpose areas, playgrounds, water features, gardens， walking paths, built structures

**Ordinal Scale:**
- `shade_along_paths`: None / Some (<50%) / Abundant (>50%)

**Data Structure:** One row per (rater × image) with normalized snake_case fields, ISO8601 timestamps, and anonymized rater codes.

## CNN Development
This repository contains the machine learning pipeline to:
1. Process survey response data into training labels
2. Train multi-task CNN models on satellite images
3. Predict green space characteristics at scale
4. Validate model performance against human annotations

## DL Choice: Tensorflow - Keras Integration
**Why Tensorflow**
1. Keras API
High-level Tensorflow API. It takes care of the training loop, metrics, logging, and saving models.
- Concise and User-friendly
2. Multi-task models
Keras makes it straightforward to build one CNN with multiple outputs. It suits our usecase because:
- we have multilabel binary, ordinal, and 1-5 features. Keras can have a binary head for features, a categorical head for shade, and a 5-class head for structure. 

**Our Label:**
- Multi-label binary: sports_field, multipurpose_open_area, childrens_playground, water_feature, gardens, walking_paths (Round 5)
- Ordinal: shade_along_paths ∈ {None, Some (<50%), Abundant (>50%)}.
- Global 1–5 score: structured_unstructured_rating
