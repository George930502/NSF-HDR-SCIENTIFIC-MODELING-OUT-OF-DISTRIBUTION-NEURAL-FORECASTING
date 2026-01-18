# Neural Forecasting Task – End-to-End Development Prompt

You are an AI engineer tasked with building a complete, Codabench-compatible neural forecasting solution for the **HDR Challenge Year 2 – Neural Forecasting** task.

---

## 1. Grounding and Repository Understanding

- **Deeply read and follow the task description** in `task.md`. Treat it as the single source of truth for:
  - prediction targets  
  - input / output formats  
  - evaluation protocol  
  - submission constraints and scoring rules  

- **Thoroughly inspect** the files under `HDRChallenge_y2/NeuralForecasting/` to understand:
  - existing baselines and utilities  
  - expected model interfaces  
  - training, inference, and packaging scripts  
  - submission guidelines
  - Development phase on codabench utilizes ```ingestion.py``` to perdict and ```scoring.py``` to calculate the final score

---

## 2. Paper-Driven Improvements (Using MCP Tools)

- Search papers online and identify the most relevant for the task

- Prioritize papers that provide **actionable improvements** for this task

- **Download the selected papers** and store them under the directory ```/references```

- Read the papers and extract key insights, including:
  - methods compatible with data shape
  - recommended model architectures
  - preprocessing and normalization strategies
  - loss functions suitable for forecasting
  - domain-specific insights for μECoG signals

---

## 3. End-to-End Forecasting System

Implement a full neural forecasting pipeline under ```/develop```

This pipeline must include:
- **deeply** analyze the properties of the raw dataset
- dataset loading and preprocessing (**preprocessing is extrmemly important for the task**)
- model definition
- training loop and checkpointing
- inference pipeline compatible with Codabench
- reproducibility controls (random seeds, logging, deterministic behavior when possible)

---

## 4. Codabench Compatibility and Evaluation

- Codabench evaluates submissions using the testing data located at:
  ```
  dataset/test
  ```

- After development:
  - run a **full local evaluation** using the testing data
  - verify that outputs strictly follow the submission format defined in `task.md`
  - ensure the submission can run in a clean environment (no hidden dependencies, correct entry points)

---

## 5. Development Environment
- create a virtual environment and install the packages using ```uv``` command
- use ```GPU``` to accelerate the process (the device has cuda support)
- use up-to-date package to develop the model

---

**Important:**
Dynamic and reflective problem-solving through thought sequences.  
Always prioritize correctness and Codabench compatibility. Performance improvements must not break the required interfaces or evaluation pipeline.
Push the code to github with project named NSF-HDR-SCIENTIFIC-MODELING-OUT-OF-DISTRIBUTION-NEURAL-FORECASTING.