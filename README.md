# agentic-AI-Research-on-crop-disease
Explainable Federated Agentic AI
for Sustainable Precision Agriculture
Multitask Learning · Calibration · GPU-Optimized Deployment
Ahtezaz Ahsan · Amna Shahid · Zoha Wajahat
School of Computing, FAST-NUCES, Islamabad, Pakistan
IEEE ACCESS | 2026
98.65% Federated Accuracy	0.0101 Best ECE (Calibration)	96.61% Weed Detection	RTX 4090 GPU Platform


Abstract
We present a novel Explainable Federated Agentic AI framework designed for sustainable precision agriculture. Our system integrates Federated Learning (FL) with multitask learning, explainability, and calibration diagnostics to address data privacy barriers in multi-farm AI deployment. The framework employs MobileNetV2 and DenseNet121 as dual encoders for disease classification and EfficientDet-D0 for weed detection, achieving a federated global accuracy of 98.65% on tomato leaf disease classification and 96.61% on weed detection. Explainability is provided through Grad-CAM spatial attribution, Kernel SHAP additive feature importance, and LIME local linear surrogates. The system is validated with Expected Calibration Error (ECE) diagnostics and optimized for NVIDIA RTX 4090 hardware using PyTorch AMP and TF32 precision.

Keywords
Federated Learning, Precision Agriculture, Multitask Learning, Explainable AI, Grad-CAM, SHAP, LIME, Calibration, FedAvg, DenseNet121, EfficientDet, Privacy-Preserving AI

1.  Introduction & Motivation
The global food security challenge demands smarter, resource-efficient crop management solutions. Rising input costs and tightening environmental regulations make traditional, labor-intensive inspection methods inadequate. Yet artificial intelligence approaches that rely on centralized training face critical barriers in agricultural settings.

Three core problems motivate this work:

•	Rising input costs and tightening environmental regulations demand smarter, resource-efficient crop management strategies.Global Food Security: 
•	Farmers are unwilling to share raw canopy images. Centralized databases are impractical for multi-farm AI deployments.Data Privacy Barriers: 
•	Heterogeneous datasets and limited connectivity motivate the need for lightweight yet expressive encoders.Rural Bandwidth Limits: 

Our framework addresses these challenges through five complementary capabilities:

•	Gradient sharing instead of raw pixel transmission, ensuring privacy.Federated Learning: 
•	Simultaneous disease classification and weed detection from shared representations.Multitask Learning: 
•	Grad-CAM, SHAP, and LIME for field-deployable interpretability.Explainable AI: 
•	Expected Calibration Error (ECE) measurement for deployment reliability.Calibration Diagnostics: 
•	Automatic Mixed Precision (AMP) and TF32 on NVIDIA RTX 4090.GPU-Optimized Inference: 

2.  System Architecture: Federated Agentic Pipeline
The proposed framework follows a four-stage federated agentic pipeline, where each stage corresponds to a functional module of the distributed AI system:

Stage 1: Perception
MobileNetV2 and DenseNet121 serve as dual encoders processing 224×224 RGB canopy images. EfficientDet-D0 handles 512×512 weed detection tiles using focal loss to address class imbalance. Each encoder extracts hierarchical visual features without transmitting raw image data.
Stage 2: Collaborative Optimization
K=2 federated clients train independently on private non-IID data shards, executing 5 local epochs per communication round over T=5 FedAvg rounds. Only model gradient updates are transmitted across the network; no raw pixel data leaves the farm environment.
Stage 3: Global Semantics
The DenseNet121 global model aggregates client updates via FedAvg (Equation 7). Dual-task heads are maintained: a pathology classification branch (11 tomato disease classes) and a weed localization branch (binary crop vs. weed detection).
Stage 4: Governance & Explainability
A comprehensive XAI and governance layer is applied at inference time, generating Grad-CAM spatial heatmaps, Kernel SHAP feature attribution maps, and LIME surrogate explanations. ECE calibration diagnostics ensure probability fidelity for field-deployable audit trails.

3.  Datasets & Experimental Setup

3.1  Tomato Leaf Disease Dataset (Kaggle)
Parameter	Value
Classes	11 (10 diseases + healthy)
Training Tiles	25,851 images
Validation Tiles	3,341 images (seed 42)
Test Tiles	3,342 images (seed 42)
Input Resolution	224 × 224 RGB
Augmentation	Horizontal Flip + ImageNet Normalization

3.2  WeedCrop Dataset (Kaggle, YOLO Format)
Parameter	Value
Classes	Binary: crop vs. weed
Training Tiles	2,368 images
Validation Tiles	235 images
Test Tiles	118 images
Input Resolution	512 × 512 RGB
Loss Function	Focal Loss (α=1, γ=2)

3.3  Hardware & Hyperparameters
Setting	Value
GPU	NVIDIA RTX 4090
Framework	Python 3.12, PyTorch AMP + TF32
Optimizer	Adam (β₁=0.9, β₂=0.999, WD=10⁻⁴)
Reproducibility	Seed = 42
Local Epochs	12 epochs (local training)
FL Rounds	T=5 FedAvg rounds, E=5 local epochs/round

4.  Federated Multitask Model Design
The core model is built on a DenseNet121 shared convolutional encoder with two specialized task heads:

4.1  Shared Backbone: DenseNet121
DenseNet121 is selected for its dense feature reuse property, making it highly parameter-efficient under non-IID federated data distributions. The shared encoder processes 224×224 inputs and extracts hierarchical feature maps transmitted only as model weights during federation.
4.2  Classification Head (Tomato Disease)
The classification branch uses Global Average Pooling (GAP) → Fully Connected layer → Softmax output over C=11 tomato disease classes. Dropout (rate=0.4) is applied on FC layers to regularize against non-IID overfitting. Loss function: Cross-Entropy (Equation 2).
4.3  Detection-Style Head (Weed Localization)
The detection head uses a convolutional layer producing bounding box offsets and class logits for binary weed vs. crop classification. Combined loss function: CIoU + Binary Cross-Entropy (Equation 5), with Focal Loss (α=1, γ=2) to handle class imbalance in the WeedCrop dataset.
4.4  FedAvg Aggregation (Equation 7)
The global model is aggregated using the weighted FedAvg formula:
θ⁽ᵗ⁺¹⁾  =  Σ [ nₖ / Σnⱼ ] · θₖ⁽ᵗ’ᴸ⁾
Non-IID partitions (K=2 clients) ensure that federated aggregation must generalize across heterogeneous farm data distributions, simulating realistic deployment conditions.

5.  Classification Results: Accuracy & F1 Score

5.1  Tomato Disease Classification
Model	Accuracy	Precision	Recall	F1 Score
MobileNetV2	97.94%	97.88%	98.17%	98.01%
DenseNet121 (Local)	98.03%	98.14%	98.22%	98.17%
Global FL (DenseNet121) ✔	98.65%	98.79%	98.75%	98.77%

The federated global model (98.65%) outperforms both local specialists, demonstrating that federated aggregation across heterogeneous non-IID farm data improves generalization.

5.2  Weed Detection (EfficientDet-D0)
Accuracy	Precision	Recall	F1 Score
96.61%	98.17%	84.62%	89.97%

The high precision (98.17%) with moderate recall (84.62%) reflects a conservative detection policy suitable for herbicide recommendation workflows, where false positives carry significant economic cost.

6.  Explainability Layer: Grad-CAM, SHAP & LIME
A triple explainability stack is applied to provide complementary perspectives on model decision-making:

6.1  Grad-CAM: Spatial Attribution
Grad-CAM weights the final convolutional channels by gradient magnitude of the target class score, generating class-discriminative spatial heatmaps at input resolution. The formulation is:
αᶜₖ = (1/HW) Σᵢⱼ ∂zᶜ/∂Aᵏᵢⱼ
•	MobileNetV2: Edge-focused activation patterns emphasizing leaf boundary features.
•	DenseNet121: Lesion-interior focused patterns capturing disease core textures.
•	Output: Heatmaps archived alongside logits for insurer-grade advisory audit trails.

6.2  Kernel SHAP: Additive Feature Importance
Kernel SHAP approximates global Shapley contributions via weighted linear regression on masked superpixel partitions. Key diagnostic capabilities include:
•	Multi-checkpoint montage for cross-architecture comparison across MobileNetV2 and DenseNet121.
•	Highlights color skew, vein patterns, and sporulation density as top discriminative features.
•	Diverging rankings flag high-risk edge cases near class decision boundaries.

6.3  LIME: Local Linear Surrogates
LIME fits sparse linear models on perturbed superpixel partitions, providing locally faithful explanations independent of the global model structure:
•	Positive weights identify scorch, mosaic, and mildew textures as disease indicators.
•	Negative weights suppress soil background and glare artifacts.
•	LIME + Grad-CAM agreement increases agronomist reviewer confidence for advisory sign-off.

7.  Calibration & Reliability Diagnostics
Expected Calibration Error (ECE) measures the alignment between predicted confidence and empirical accuracy across 10 equal-width probability bins:
ECE  =  Σₘ |Bₘ|/N · |acc(Bₘ) − conf(Bₘ)|

Improving ECE Trend	Weed Detection ECE	Deployment Threshold
MobileNetV2: 0.0117 DenseNet121: 0.0105 Federated Global: 0.0101 Federation tightens probability fidelity.	ECE = 0.0587 Higher ECE from focal loss and binary class imbalance. Conservative probabilities. Do NOT compare directly with tomato ECE.	Tomato ECE < 0.012 across all model paths. Safe for automated sprinkler and scouting threshold triggers.

8.  Comparison with Prior Work
Study	FL	Multitask	XAI	Calibration	Non-IID
Aggarwal et al. (rice FL)	✓	✗	✗	✗	✓
Hari et al. (plant leaves)	✓	∼	✗	✗	∼
Hassan & Maji (plant disease)	✗	✗	✗	✗	✗
Nagasubramanian et al. (UQ)	✗	✗	✗	✓	✗
Karimireddy (SCAFFOLD)	✓	✗	✗	✗	✓
Srinivasu [6] (baseline)	∼	∼	∼	✗	✗
Ours (this paper) ★	✓	✓	✓	✓	✓

✓ = Yes  |  ✗ = No  |  ∼ = Partial  |  FL = Federated Learning  |  XAI = Explainability  |  Cal. = Calibration Reporting

Our framework is the only work in the surveyed literature to simultaneously satisfy all five capability dimensions: federated learning, multitask learning, explainability, calibration reporting, and non-IID data handling, on agronomic data.

9.  Conclusion & Future Work

9.1  Key Contributions
•	Federated DenseNet121 achieves 98.65% accuracy, surpassing local MobileNetV2 (97.94%) and DenseNet121 (98.03%) and exceeding the Srinivasu [6] baseline by approximately 3–6%.
•	EfficientDet-D0 weed detection branch achieves 96.61% accuracy with high precision (98.17%), optimized for conservative herbicide recommendation workflows.
•	Triple explainability stack (Grad-CAM + SHAP + LIME) combined with ECE < 0.012 on the tomato pathway ensures trustworthy, auditable decisions suitable for regulated advisory programs.
•	Privacy is fully preserved: only model tensors are transmitted. Raw canopy imagery never leaves the farm silo during federated training.

9.2  Future Research Directions
•	Extend FedAvg rounds and add personalized FL layers for highly skewed, small-farm distributions.
•	Apply hard-negative mining to improve weed detection recall (currently 84.62% under focal loss).
•	Conduct prospective field trials with extension services and agronomists-in-the-loop evaluation.
•	Implement temperature scaling and ensemble calibration methods for on-device deployment.
•	Integrate differential privacy budgets and secure aggregation protocols for production readiness.


References
1.	Aggarwal, S. et al. (2023). Federated learning for rice disease classification in non-IID agricultural settings.
2.	Hari, R. et al. (2022). Federated deep learning for plant leaf disease diagnosis.
3.	Hassan, A. & Maji, A. K. (2023). Plant disease detection using deep learning without federated setting.
4.	Nagasubramanian, K. et al. (2022). Uncertainty quantification in crop phenotyping with deep neural networks.
5.	Karimireddy, S. P. et al. (2020). SCAFFOLD: Stochastic controlled averaging for federated learning. ICML.
6.	Srinivasu, P. N. et al. (2022). Classification of skin disease using deep learning neural networks with MobileNet V2 and LSTM. Sensors.
7.	Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV.
8.	Lundberg, S. M. & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
9.	Ribeiro, M. T. et al. (2016). Why should I trust you? Explaining the predictions of any classifier. KDD.
10.	McMahan, H. B. et al. (2017). Communication-efficient learning of deep networks from decentralized data. AISTATS.
