# Technical Report: Point-Supervised Remote Sensing Segmentation

## 1. Method

### 1.1 Partial Cross-Entropy Loss

Traditional semantic segmentation requires dense pixel-level annotations, which are expensive and time-consuming to collect. We implement **Partial Cross-Entropy Loss** to enable training with sparse point annotations.

**Mathematical Formulation:**

Given predictions $P \in \mathbb{R}^{C \times H \times W}$ and sparse labels $Y \in \mathbb{R}^{H \times W}$ where most pixels are unlabeled (denoted by $ignore\_index = -1$):

$$
L_{partial} = -\frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} \log p_{y_{ij}}^{ij}
$$

Where:
- $\Omega = \{(i,j) | y_{ij} \neq -1\}$ is the set of labeled pixels
- $p_c^{ij}$ is the predicted probability for class $c$ at position $(i,j)$
- $|\Omega|$ is the number of labeled pixels

**Key Advantages:**
- Reduces annotation cost by 95-99%
- Enables training with minimal supervision
- Maintains spatial context through the segmentation network

### 1.2 Point Label Sampling Strategies

We implement multiple sampling strategies to simulate different annotation scenarios:

1. **Random Sampling**: Uniform random selection across the image
2. **Uniform Grid**: Evenly distributed spatial coverage
3. **Class-Balanced**: Equal representation of all classes
4. **Boundary-Aware**: Focus on class boundaries (harder regions)
5. **Cluster Sampling**: Points in spatial clusters (mimics human annotation)

### 1.3 Network Architecture

We use U-Net with ResNet34 encoder (pre-trained on ImageNet):
- **Encoder**: ResNet34 (downsampling path)
- **Decoder**: Upsampling with skip connections
- **Output**: Pixel-wise classification for C classes

---

## 2. Experimental Design

### Research Questions:

**Q1: How does the number of point annotations affect segmentation performance?**

**Q2: How do different sampling strategies impact model accuracy?**

### 2.1 Experiment 1: Effect of Number of Points

**Hypothesis**: Segmentation performance improves with more point annotations, but with diminishing returns after a threshold.

**Experimental Setup:**
- **Variable**: Number of points per image: [50, 100, 200, 500, 1000, 2000]
- **Fixed**: Sampling strategy (uniform), network architecture, training hyperparameters
- **Metric**: Mean IoU (mIoU) on validation set with full annotations
- **Baseline**: Fully supervised training (100% pixels labeled)

**Expected Results:**
- Performance increases with more points
- Diminishing returns beyond 500-1000 points
- Gap between point-supervised and fully-supervised narrows

**Procedure:**
```python
num_points_list = [50, 100, 200, 500, 1000, 2000]
results = {}

for num_points in num_points_list:
    # Initialize sampler
    sampler = PointLabelSampler(num_points=num_points, strategy='uniform')
    
    # Train model
    model = train_model(sampler=sampler, epochs=50)
    
    # Evaluate
    miou = evaluate_model(model, val_loader)
    results[num_points] = miou
    
    print(f"Points: {num_points} | mIoU: {miou:.4f}")
```

---

### 2.2 Experiment 2: Effect of Sampling Strategy

**Hypothesis**: Sampling strategies that focus on class boundaries or ensure class balance will outperform random sampling.

**Experimental Setup:**
- **Variable**: Sampling strategy: [random, uniform, balanced, boundary, cluster]
- **Fixed**: Number of points (200), network architecture, training hyperparameters
- **Metric**: Mean IoU on validation set
- **Analysis**: Per-class IoU to understand which strategies help rare classes

**Expected Results:**
- Boundary-aware sampling performs best (focuses on hard regions)
- Class-balanced sampling helps with class imbalance
- Random sampling performs worst
- Cluster sampling may underperform due to spatial bias

**Procedure:**
```python
strategies = ['random', 'uniform', 'balanced', 'boundary', 'cluster']
results = {}

for strategy in strategies:
    # Initialize sampler
    sampler = PointLabelSampler(num_points=200, strategy=strategy)
    
    # Train model
    model = train_model(sampler=sampler, epochs=50)
    
    # Evaluate
    miou, per_class_iou = evaluate_model(model, val_loader, return_per_class=True)
    results[strategy] = {
        'miou': miou,
        'per_class': per_class_iou
    }
    
    print(f"Strategy: {strategy} | mIoU: {miou:.4f}")
```

---

## 6. References

1. Bearman, A., et al. (2016). "What's the point: Semantic segmentation with point supervision." ECCV 2016.
2. Ronneberger, O., et al. (2015). "U-net: Convolutional networks for biomedical image segmentation." MICCAI 2015.

