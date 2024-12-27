# CIFAR-10 Classification Using AI Models

## Project Overview

This project explores the implementation and evaluation of multiple AI models for image classification using the **CIFAR-10** dataset. The models implemented include:

- **Naive Bayes Classifier**
- **Decision Tree Classifier**
- **Multi-Layer Perceptron (MLP)**
- **Convolutional Neural Network (CNN)** - Based on the **VGG11** architecture

The primary goal of the project is to analyze and compare traditional machine learning techniques with modern deep learning approaches, identifying strengths, limitations, and practical applications.

---

## Dataset Description

The **CIFAR-10** dataset consists of **60,000 32x32 RGB images** evenly distributed across **10 classes**, including:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

### Dataset Challenges:

1. **High Dimensionality**: Each image contains 3072 features (32×32×3).
2. **Low Resolution**: Small image size limits distinguishable details.
3. **Class Similarity**: Overlapping features between classes, such as cats and dogs, lead to potential misclassifications.

---

## Methodology

### 1. Preprocessing

- **Resizing and Normalization**: Images resized to 224×224 and normalized for consistency with pre-trained models.
- **Feature Extraction**: Leveraged **ResNet-18** for generating 512-dimensional feature vectors.
- **Dimensionality Reduction**: Applied **PCA** to reduce features to 50 dimensions for efficiency.
- **Train-Test Split**: Balanced subsets of 500 training images and 100 test images per class.

### 2. Model Implementations

#### A. Naive Bayes Classifier

- **Approach**: Probabilistic classification assuming feature independence.
- **Evaluation**:
  - Accuracy: **77%**
  - Precision and Recall: Struggled with visually similar classes.
- **Strengths**: Computational efficiency and simplicity.
- **Limitations**: Inability to capture complex relationships between features.

#### B. Decision Tree Classifier

- **Approach**: Tree-based recursive partitioning using Gini impurity.
- **Evaluation**:
  - Accuracy: **60.3%** (optimal depth: 10).
  - Overfitting beyond 10 levels of depth.
- **Strengths**: Interpretability and non-linear decision boundaries.
- **Limitations**: Sensitive to depth and prone to overfitting.

#### C. Multi-Layer Perceptron (MLP)

- **Architecture**:
  - Input: 50-dimensional vectors
  - Hidden Layers: 3 layers with ReLU activation and Batch Normalization.
  - Output: Softmax activation for classification.
- **Evaluation**:
  - Accuracy: **82.6%**
  - Performance stabilized at depth 3.
- **Strengths**: Generalization ability and adaptability.
- **Limitations**: Lack of spatial feature learning compared to CNNs.

#### D. Convolutional Neural Network (CNN)

- **Architecture**: VGG11-inspired model optimized for CIFAR-10.
- **Training Details**:
  - SGD optimizer with momentum.
  - Evaluated across kernel sizes: **2×2**, **3×3**, **5×5**, and **7×7**.
  - Accuracy: **83.2%**
- **Strengths**:
  - Captures spatial hierarchies effectively.
  - Best performance among all models.
- **Limitations**:
  - Overfitting with deeper layers.
  - Struggled with visually similar classes (e.g., cats vs. dogs).

---

## Evaluation Metrics

The models were evaluated based on:

- **Accuracy**: Overall correctness of predictions.
- **Precision, Recall, and F1-Score**: Class-specific performance indicators.
- **Confusion Matrices**: Visual representation of misclassifications.

### Key Observations

- Naive Bayes was computationally efficient but struggled with overlapping features.
- Decision Trees handled non-linear patterns but faced overfitting challenges.
- MLP provided a balance between complexity and generalization.
- CNNs excelled in capturing intricate patterns but required careful regularization to avoid overfitting.

---

## Results Summary

| Model                        | Accuracy (%) | Strengths                                      | Limitations                                   |
| ---------------------------- | ------------ | ---------------------------------------------- | --------------------------------------------- |
| Naive Bayes                  | 77.0         | Simple, fast, interpretable                    | Assumes feature independence, low performance |
| Decision Tree                | 60.3         | Flexible, interpretable                        | Sensitive to depth, prone to overfitting      |
| Multi-Layer Perceptron       | 82.6         | Generalizes well, balances complexity          | Lacks spatial feature learning for image data |
| Convolutional Neural Network | 83.2         | Captures spatial hierarchies, best performance | Overfitting, requires careful tuning          |

---

## Key Insights and Recommendations

1. **Feature Engineering Matters**:

   - Using ResNet-18 for feature extraction significantly improved performance.
   - PCA reduced computational costs while preserving important features.

2. **Model Selection Depends on Use Case**:

   - Simpler models (Naive Bayes, Decision Trees) work well for quick baselines.
   - Neural networks (MLP, CNN) provide superior performance but demand higher resources.

3. **Regularization and Hyperparameter Tuning**:

   - Overfitting in CNNs can be reduced by data augmentation, dropout, and weight decay.
   - MLP depth and hidden layer sizes must be tuned to avoid diminishing returns.

4. **Advanced Architectures for Future Work**:

   - Testing architectures like **ResNet** and **Transformers** may yield higher accuracy.
   - Exploring transfer learning and ensemble techniques can further optimize results.

---

## Contributors

- **Maharaj Teertha Deb** - Student ID: 40227747 (Myself)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **PyTorch** and **Scikit-learn** for libraries and pre-built functions.
- **CIFAR-10 Dataset** for benchmarking.
- **ResNet-18** for feature extraction support.

---

## Contact

For any questions or collaboration requests, please reach out via:

- **Email**: [maharaj.deb.concordia@gmail.com](mailto\:maharaj.deb.concordia@gmail.com)
- **GitHub**: [GitHub Profile/Teertha Deb](https://github.com/TeerthaDeb)

---

## Try it Yourself

All required files, including sample notebooks and a PDF report, are provided in the repository. You can run the experiments directly via [Google Colab](https://colab.research.google.com/drive/17-eU0sA3gvnFQiaDmDIX2IJPLCY8CTRa#scrollTo=wywRjIouwo8L) or Jupyter Notebook.