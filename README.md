# Colony Counter

This project provides a complete training and evaluation workflow for automatic colony counting using deep learning.  
It supports multiple neural architectures, enabling flexible model selection.

---

## 0.Environment Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate colony
```

## 1.How to use the repository
### MODEL_DICTIONARY – Model Selection System

The project uses a central `MODEL_DICTIONARY` to register all available neural network architectures.  
Each entry defines:

- the **model class**  
- the **constructor arguments** (`kwargs`)  
- the **default weight file** to load/save  

This makes it possible to select any model using a simple CLI flag like:
```bash
--model EfficientNet
```

To add a new model, simply insert another entry:

```
"my_new_model": {
    "class": MyModelClass,
    "kwargs": {"hidden_dim": 512},
    "weights": "my_new_model.pth"
}
```
After registering the model, you can immediately use it for:
- training :

```
python src/ml/training.py --model my_new_model
```

- evaluation :

```
python src/ml/evaluate.py --model my_new_model
```
