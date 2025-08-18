# GreenSpace_CNN

## Overview
Train a CNN to reproduce green-space audits at scale using human judgments from photos/satellite images.

## Survey Design
Human raters evaluate images across standardized criteria:

**Core Rating:**
- `structured_unstructured_rating`: 1-5 scale (1=very structured, 5=very unstructured)

**Binary Presence Items:** (Yes/No/Can't answer)
- Sports fields, multipurpose areas, playgrounds, water features, gardens
- Walking paths, picnic areas, infrastructure, public art, artificial canopies

**Ordinal Scale:**
- `shade_along_paths`: None / Some (<50%) / Abundant (>50%)

**Data Structure:** One row per (rater × image) with normalized snake_case fields, ISO8601 timestamps, and anonymized rater codes.

## CNN Development
This repository contains the machine learning pipeline to:
1. Process survey response data into training labels
2. Train multi-task CNN models on satellite/aerial imagery
3. Predict green space characteristics at scale
4. Validate model performance against human annotations

