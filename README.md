# ALL IN ONE VISION 4 (A4)

**ALL-IN-ONE-VISION is a collection of self-contained projects, organized inside the `projects/` folder.**

The core idea is that **every project is an edited instance of a general project template**. The template is the **primary product** of the repository; individual projects are its task- and dataset-specific implementations.

The template provides a standardized, low-code structure and workflow that can be adapted to different computer vision tasks, datasets, and experiments without requiring users to build each project from scratch.

**A4** is the YOLO-focused implementation of this concept. It is a **simple, low-code, experiment-oriented repository**, built around **Jupyter Notebooks and configuration**, for **fine-tuning and transfer learning of YOLO models** using the **Ultralytics Package and Platform** on GPU environments such as **Google Colab and Kaggle**.

A4 is deliberately built around the same principle as the overall ALL-IN-ONE-VISION architecture: **start from the general project template, adapt only what is necessary, and delegate core ML functionality to mature external tools rather than reinventing it.**

## **Core Principle**

<aside>

### **1. Good-Enough, Bottleneck-Driven, Extremely Low-Code & Simple**

A4 is **not intended to be a highly advanced or massive ML framework**. Its purpose is to provide a simple, practical workflow for experimentation with Ultralytics.

Our priorities are:

- **Simplicity of code**
- **Ease of use**
- **Fast experimentation**
- **Minimal custom implementation**

A4 should address **practical bottlenecks that actually matter** rather than adding complexity for its own sake.

We deliberately avoid extensive abstractions, unnecessary configuration, and custom implementations of functionality that Ultralytics already provides.

</aside>

<aside>

### **2. Build on Mature Software**

Ultralytics already handles the core YOLO functionality, including **training, validation, inference, and deployment**.

A4 should therefore **build on top of the Ultralytics ecosystem rather than reimplement it**.

Custom source code should be kept **as small as reasonably possible**, with A4 focusing on the parts that Ultralytics does not already solve: providing a **simple, practical, experiment-oriented workflow** around the existing ecosystem.

</aside>

<aside>

### **3. Standard First, Zero Reinvention**

**Use Ultralytics before building A4.**

Follow Ultralytics’ **official documentation, repositories, and notebooks** as the primary source of truth. Before designing, configuring, or implementing something, first check whether Ultralytics already provides a standard solution.

- **Low-code by default.** If substantial custom code is required, first verify whether Ultralytics already provides the functionality.
- **Standard solutions first.** For common problems, look for documented or established solutions before creating our own.
- **Understand before abstracting.** When something appears to require extensive design or reasoning, first make sure the existing Ultralytics workflow and conventions are fully understood.

**If Ultralytics already solves a problem well, A4 should use it—not reinvent it.**

#### Search-first

- [Ultralytics Official Website](https://www.ultralytics.com/)
- [Ultralytics Docs](https://docs.ultralytics.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [Ultralytics Notebooks](https://github.com/ultralytics/notebooks)
- [Ultralytics Academy](https://academy.ultralytics.com/)
- [Ultralytics Platform](https://platform.ultralytics.com/)
- [Ultralytics Community Forum](https://community.ultralytics.com/)
</aside>

## Design

<aside>

### **Project Structure : Project Template as the Primary Product**

The fundamental unit of ALL-IN-ONE-VISION is not an individual project—it is the **general project template**.

Every project inside `projects/` is created by taking this template and adapting it to a specific **task, dataset, and set of experiments**.

Conceptually:

```jsx
ALL-IN-ONE-VISION-4/
│
├── template/                    # General project template — primary product
│   ├── configs/
│   │   ├── datasets/
│   │   └── experiments/
│   ├── datasets/
│   │   └── README.md
│   ├── docs/
│   ├── notebook.ipynb           # Primary experiment interface
│   ├── runs/                    # Ultralytics-generated outputs
│   │   └── README.md
│   ├── scripts/
│   │   ├── data/
│   │   └── inference/
│   └── README.md
│
├── projects/                    # Task- and dataset-specific template instances
│   ├── Classify/
│   ├── Depth/
│   ├── Detect/
│   │   └── VOC/
│   │       ├── configs/
│   │       │   ├── datasets/
│   │       │   │   └── VOC.yaml
│   │       │   └── experiments/
│   │       ├── datasets/
│   │       │   └── README.md
│   │       ├── docs/
│   │       ├── notebook.ipynb
│   │       ├── runs/
│   │       │   └── README.md
│   │       ├── scripts/
│   │       │   ├── data/
│   │       │   └── inference/
│   │       └── README.md
│   ├── OBB/
│   ├── Pose/
│   ├── Segment/
│   ├── Semantic/
│   └── README.md
│
├── docs/                        # Repository-level documentation
│
└── README.md
```

The template is designed to be **model-, dataset-, and task-agnostic** wherever practical. A project inherits its structure and workflow from the template and modifies only the parts necessary for its specific use case.

For example:

```
template/
    ↓
adapt to detection + Pascal VOC
    ↓
projects/Detect/VOC/
```

</aside>

<aside>

### **Task & Dataset–Based Project Organization**

**Projects are organized by task and dataset, not by model.**

The top-level directories represent the **task**, while each project represents a specific **task + dataset** combination.

For example:

```jsx
projects/
└── Detect/
		└── VOC/
```

A project can contain **many experiments** using different models, hyperparameters, or configurations. These experiments belong to the same project and **do not require separate projects**.

For example, `Detect/VOC` may contain experiments using YOLO11n, YOLO26n, or different configurations of the same model.

The goal is to keep A4 **task- and dataset-oriented rather than model-oriented**, while making model comparison and experimentation straightforward.

</aside>

<aside>

### **`template/` & `template/notebook.ipynb`**

Every project is an **edited instance of the general project template**. 

Engineers start from the template and adapt its configuration, data, and designated sections to the specific **task, dataset, and experiments**.

The template’s primary interface is a **single Jupyter Notebook** designed as a hands-on, step-by-step workflow.

Engineers should primarily:

- Fill in predefined sections
- Select from provided options
- Configure YAML files when necessary
- Run and inspect the workflow
- Adapt only designated areas when customization is genuinely required

Custom code should be **rare and limited to explicitly designated areas** where the standard workflow is insufficient.

The ultimate goal is to **minimize the amount of code engineers and users need to write**, while keeping the workflow simple, reproducible, and easy to understand.

</aside>

<aside>

### **Notebook-Centered Projects**

Each project is built around a **single Jupyter Notebook that serves as the experiment’s main interface, workflow, and documentation**.

The notebook is a **guided, executable experiment**, rather than a collection of independent code examples.

It leads the user through the complete workflow:

**1. Experiment Setup**

Define the experiment, select the model and dataset, configure the environment, and set only the parameters that need to be changed.

**2. Dataset Preparation & Validation**

Load or locate the dataset, prepare it when necessary, verify its structure and annotations, inspect class distributions, and visualize representative samples before training.

**3. Model & Training**

Load the selected pretrained model and run training using Ultralytics, with project-specific configuration and only purposeful overrides to its defaults.

**4. Evaluation & Inspection**

Evaluate the trained model, inspect metrics and training results, visualize predictions, and examine representative successes and failures.

**5. Experiment Summary**

Collect the important configuration, metrics, artifacts, observations, and conclusions into a concise record of the experiment.

**6. Export**

Export the selected trained model to the required deployment format using Ultralytics.

**7. Report**

Generate a reproducible experiment report containing the dataset, configuration, training results, evaluation, exported artifacts, and key findings.

The notebook should contain **predefined cells, explanations, visualizations, and configuration points**, allowing users to primarily **run, configure, inspect, and adapt** the workflow rather than write the underlying ML code.

A4 keeps **project-specific data preparation, inspection, experiment logic, and reporting** in the notebook while delegating model training, evaluation, inference, and export to the Ultralytics ecosystem.

The result should be a **self-contained, low-code experiment notebook** that is easy to understand, run, inspect, reproduce, and extend—without introducing unnecessary scripts or custom frameworks.

</aside>

<aside>

### **Default-First Configuration & Selective Customization**

**Ultralytics already provides sensible defaults, so A4 should not reconfigure what does not need to be configured.**

Configurations should contain only the parameters that we have a **strong reason to change**—for example, settings that are necessary for the project or are expected to provide a meaningful improvement over the baseline.

The default Ultralytics behavior should remain the **starting point and baseline**. Configuration should be **minimal and intentional**, rather than an attempt to expose or reproduce every available option.

> **If a setting does not need to be changed, leave it to Ultralytics.**
> 

> **YAML files define overrides, not a complete duplicate of Ultralytics’ configuration.**
> 
</aside>

<aside>

### **Automated Tuning with Ultralytics[#](https://www.ultralytics.com/glossary/hyperparameter-tuning#automated-tuning-with-ultralytics)**

The `ultralytics` library simplifies optimization by including a built-in [tuner](https://docs.ultralytics.com/reference/engine/tuner) that utilizes **genetic algorithms**.  

- Example
    
    The tuner will mutate hyperparameters over several iterations to maximize [Mean Average Precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map).
    
    ```
    from ultralytics import YOLO
    
    # Initialize a YOLO26 model 
    model = YOLO("yolo26n.pt")
    
    # Start tuning hyperparameters on the COCO8 dataset
    # The tuner runs for 30 epochs per iteration, evolving parameters like lr0 and momentum
    model.tune(data="coco8.yaml", epochs=30, iterations=100, optimizer="AdamW", plots=False)
    ```
    
- **Preparing for Hyperparameter Tuning[#](https://docs.ultralytics.com/guides/hyperparameter-tuning#preparing-for-hyperparameter-tuning)**
    
    Before you begin the tuning process, it's important to:
    
    1. **Identify the Metrics**: The built-in tuner (`use_ray=False`) ranks trials by the task's `fitness` score, while `use_ray=True` uses the task metric. Decide which secondary metrics you will inspect, such as AP50 or F1-score.
    2. **Set the Tuning Budget**: Define how much computational resources you're willing to allocate. Hyperparameter tuning can be computationally intensive.
    3. **Use Ultralytics’ built-in tuning workflow** rather than introducing a custom optimization framework.
</aside>

<aside>

### **Ultralytics Platform & MLflow Integration**

For large-scale operations, teams can leverage the [Ultralytics Platform](https://platform.ultralytics.com/) to manage datasets and visualize tuning experiments in the cloud.

The **Ultralytics Platform** is the primary integration for project, dataset, experiment, and model management, while **Google Colab GPUs** serve as the primary training environment through A4 notebooks.

**MLflow remains optional**, used only when additional experiment tracking provides a clear benefit.

A4 should avoid building custom dashboards, tracking systems, model registries, or training-management abstractions unless existing tools cannot reasonably provide the required functionality.

</aside>

---

<aside>

### **Template Components**

Each project follows the template’s basic structure:

| **Directory / File** | **Purpose** |
| --- | --- |
| `notebook.ipynb` | Main experiment interface and executable workflow |
| `configs/datasets/` | Dataset-specific configuration |
| `configs/experiments/` | Intentional experiment overrides |
| `datasets/` |  local datasets live here |
| `runs/` | Native Ultralytics-generated experiment outputs. A4 does not create separate `artifacts/`, `checkpoints/`, `logs/`, `metrics/`, or `outputs/` directories to duplicate information already managed by Ultralytics. |
| `scripts/data/` | Minimal custom data-processing utilities when required |
| `scripts/inference/` | Minimal custom inference utilities when required |
| `docs/` | Project-specific documentation |
| `README.md` | Project overview and usage |
</aside>

---

Note1 : Do NOT throw away pretrained knowledge: Transfer Classes with Name Aliases

Note2: Focus on the **`template/notebook.ipynb.` that is the most important part of the project**