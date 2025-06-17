E-Waste Image Classification Using EfficientNetV2B0 (Transfer Learning) 

image1.png, Picture 

 

 

 

**Problem Statement **

Electronic waste (e-waste) is a rapidly growing global concern for both environmental and public health. Discarded electronic devices contain both valuable resources and hazardous substances, leading to contamination and health risks if improperly processed. Manual classification for recycling is labor-intensive, prone to errors, and inefficient. This project aims to establish an automated e-waste classification system using artificial intelligence and machine learning, enabling accurate and rapid identification to streamline recycling and promote sustainable waste management. 

 

 

 
**
 Aim 
**
The primary objective of this project is to develop a highly accurate and dependable E-Waste Image Classification model for categorizing electronic waste based on visual data. This advanced model will leverage transfer learning with the EfficientNetV2B0 algorithm, a state-of-the-art convolutional neural network. Automating e-waste identification will facilitate efficient sorting and recycling, representing a pivotal advancement towards a circular economy for electronics, promoting environmental stewardship, and safeguarding public health. 

 Learning Objectives 

The learning objectives for this project are: 

Data Understanding and Pre-processing: To meticulously prepare the E-Waste Image Dataset, ensuring data integrity, consistency, and standardized input for model processing. This includes comprehending the folder-based classification structure to accurately infer class labels. 

Exploratory Data Analysis: To visually inspect and analyze image characteristics, class distribution, and potential imbalances within the dataset. This step aids in identifying challenges like intra-class variability or inter-class similarities. 

Feature Engineering (Implicit via Transfer Learning): To understand how transfer learning implicitly leverages powerful, hierarchical pre-trained features from extensive datasets like ImageNet. This recognizes the advantages of using EfficientNetV2B0 as an efficient feature extractor, conserving training time and resources. 

Machine Learning Algorithm Application (Deep Learning with Transfer Learning): To comprehend transfer learning's benefits (reduced training time, feature exploitation, enhanced performance) and EfficientNetV2B0's operational principles (Fused MBConv blocks, progressive learning). This involves implementing and adapting the EfficientNetV2B0 architecture for e-waste classification by incorporating custom layers. 

Model Training and Evaluation: To train the developed model on the e-waste image dataset, monitoring progression with accuracy and loss metrics. This includes rigorously evaluating performance on an unseen test dataset using classification metrics like confusion matrixes and classification reports, and analyzing class-specific performance. 

 
**Model Deployment (using Gradio):** To acquire proficiency in constructing an interactive web interface using the Gradio library for model demonstration. This involves comprehending the methodology for testing the model with novel e-waste images and interpreting its predictions, providing a practical application. 

**About Project **

This project aims to develop an automated e-waste classification system using advanced deep learning techniques, specifically leveraging EfficientNetV2B0. By accurately categorizing electronic waste from images, the model will streamline sorting processes for efficient recycling. This initiative seeks to mitigate environmental and health risks associated with improper e-waste disposal. The ultimate goal is to create a reliable tool that contributes significantly to sustainable waste management and environmental protection. 

 

** Introduction to Key Concepts 

 Transfer Learning **

Transfer Learning is an efficacious deep learning methodology where a pretrained model, trained on an expansive dataset for a related task, serves as a foundational antecedent for a novel yet cognate task. This approach leverages knowledge (features, patterns, representations) from the original task, effectively transferring and adapting it. This confers substantial advantages, as training deep learning models demands considerable labeled data, computational resources, and time, often leading to overfitting with limited datasets. 

Transfer learning assuages these challenges by utilizing extensive, pre-existing knowledge embedded within models trained on colossal, generic datasets like ImageNet. These models have internalized powerful, generalizable visual features, broadly applicable across image classification tasks. This transference enables superior performance on new tasks with demonstrably reduced data and computational expenditure. 

 

 

 

**Benefits 
**
Expedited Training Regimen — Initializing models with pre-trained weights curtails the computationally intensive initial learning phase, significantly reducing resource consumption and training duration. 

Leveraging of Pre-learned Features — This approach capitalizes on robust, universal features assimilated from extensive datasets, embodying a rich hierarchy of visual patterns. This provides invaluable experience for comprehensive visual content interpretation. 

Enhanced Performance with Constrained Data — For limited task-specific datasets, transfer learning provides a resilient, pre-trained feature extraction mechanism, enabling superior generalization and mitigating overfitting. 

  

EfficientNetV2B0: Transfer Learning Backbone 

EfficientNetV2, introduced by Google in 2021, is an optimized family of convolutional neural network models. Developed through Neural Architecture Search (NAS) and progressive training, they achieve superior accuracy with reduced parameters and FLOPs. This efficiency renders them well-suited for both training and inference, particularly in resource-constrained environments or applications demanding rapid processing, such as real-time e-waste sorting. EfficientNetV2B0, the most compact variant, offers an optimal balance between performance and computational overhead. 

 

Software Requirements 

Python 3.x 

Jupyter Notebook  

 

**Tools Used 
**
1. Python 
 Used as the primary programming language for data preprocessing, exploratory data analysis, and implementation of machine learning algorithms. Python's libraries, such as numpy, matplotlib, scikit-learn, and others, facilitated efficient data manipulation, visualization, and model development. 

 

 

2. TensorFlow & Keras 

Used for constructing, training, and fine-tuning deep learning models. The EfficientNetV2B0 architecture was implemented using Keras within the TensorFlow framework. 

3. Pillow (PIL) 
 Used for loading image files and performing basic image processing operations (PIL / pillow). 

4. Gradio 
 Used the gradio library to develop user-friendly web interfaces for deploying and interactively testing machine learning models. 

5. Jupyter Notebooks 
 Used for an interactive and collaborative coding environment. Jupyter Notebooks provided a seamless platform for code execution, visualization, and documentation. 

 

** Dataset  **

Dataset Name: E-Waste Image Dataset 

Source: https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset 

 

 Findings and Insights 

      Data Exploration: 

The project utilized a folder-based image classification dataset comprising images of 10 e-waste categories: PCB, Player, Battery, Microwave, Mobile, Mouse, Printer, Television, Washing Machine, and Keyboard. 

 The dataset was organized into three directories: 
 - Train/: 2,400 images (240 images per class) 
 - Validation/: 300 images (30 images per class) 
 - Test/: 300 images (30 images per class) 

 All classes are equally balanced across the training, validation, and test sets, ensuring no class bias during model training or evaluation. 

Exploratory data analysis confirmed that each subfolder accurately represented its respective class, and sample image visualizations revealed consistent labeling and distinguishable visual features across categories. 

**Data Preprocessing: **

For image data, the preprocessing steps involved: 

Image Resizing and Rescaling: Images were uniformly resized (e.g., 128×128 pixels) and pixel values were rescaled (e.g., 0-255 to 0-1) for consistent model input and improved convergence. 

Data Augmentation: Techniques like RandomFlip, RandomRotation, and RandomZoom were applied to the training data. This expanded the dataset, enhancing feature learning and reducing overfitting. 

Image Normalization: Images were normalized using preprocess_input to match the specific input requirements of the pre-trained EfficientNet model, crucial for effective transfer learning. 

 Model Development : 

The e-waste classification model was developed using a transfer learning approach, leveraging the robust features of the pre-trained EfficientNetV2B0 neural network. This powerful model, initially trained on a massive dataset, served as the backbone, with its top classification layers removed.  

A new, custom classification head was then added, consisting of a GlobalAveragePooling2D layer to condense image features, a Dropout layer (with a 20% dropout rate) to prevent the model from memorizing the training data too closely, and a Dense layer with softmax activation to output probabilities for the 10 distinct e-waste categories. An explicit Input layer defined the expected image size of 128×128 pixels. 

The training strategy involved a crucial fine-tuning step. While the entire EfficientNetV2B0 backbone was initially set to be trainable, the first 100 layers were then specifically frozen. This allowed the model to retain the generalized low-level features learned from ImageNet, while enabling the higher-level features and the newly added classification head to adapt and learn patterns specific to the e-waste images.  

The model was then compiled using the Adam optimizer with a learning rate of 0.0001, which is a common and effective choice for deep learning. SparseCategoricalCrossentropy was chosen as the loss function, suitable for multi-class classification problems where labels are integers, and Accuracy was set as the primary metric to track performance during training. 

The model was trained for a maximum of 15 epochs, processing images in batches of 100. To ensure efficient training and prevent overfitting, an EarlyStopping callback was implemented. 

 This callback continuously monitored the val_loss (validation loss). If the validation loss did not show improvement for 3 consecutive epochs (patience=3), training would automatically stop. Furthermore, restore_best_weights=True ensured that the model's weights from the epoch with the lowest validation loss were preserved, guaranteeing that the final saved model was the best-performing version on unseen data. This comprehensive setup aimed to achieve high classification accuracy and strong generalization for the e-waste sorting task. 

Evaluation 

 Over 15 training epochs, the model exhibited rapid learning and strong performance, achieving high training accuracy (0.9908) and consistent validation accuracy (0.9600). 

The final evaluation on the unseen test dataset confirmed the model's robustness, yielding a high test accuracy of 0.9600 and a low test loss of 0.1073. Detailed analysis through the classification report and confusion matrix further demonstrated excellent precision, recall, and F1-scores across all 10 distinct e-waste categories, indicating the model's strong generalization capabilities and effectiveness in differentiating various electronic waste types. 

**Key Points derived from the classification report: **

High Overall Accuracy: The model achieved an impressive overall accuracy of 0.96 (96%), indicating its strong ability to correctly classify e-waste images. 

Excellent Performance for Specific Classes: Classes 1, 3, and 9 (likely corresponding to "Keyboard," "Mobile," and "Washing Machine" based on the example class names) achieved perfect precision, recall, and F1-scores of 1.00, demonstrating flawless classification for these categories. 

Robust Performance Across All Classes: All 10 classes show high F1-scores (ranging from 0.92 to 1.00), precision (0.88 to 1.00), and recall (0.90 to 1.00). This indicates consistent and reliable performance across the diverse e-waste categories. 

Balanced Test Set: The support of 30 samples for each class confirms that the test dataset was well-balanced, ensuring that the reported metrics are a fair representation of the model's performance across all categories. 

Strong Generalization: The high and consistent scores across all metrics and classes suggest that the model generalizes exceptionally well to unseen data, making it a reliable tool for automated e-waste classification. 

 

**Conclusion **

This laboratory exercise successfully developed an accurate and reliable E-Waste Image Classification model utilizing EfficientNetV2B0 and transfer learning techniques. The model demonstrated strong performance across both training and testing phases, achieving high accuracy and robust classification metrics for all 10 distinct e-waste categories, indicating its excellent ability to generalize to new, unseen image data. 

The project highlights the efficacy of transfer learning for image classification, especially with limited datasets, and the efficiency of the EfficientNetV2B0 architecture. The developed model, coupled with its user-friendly Gradio interface, can serve as a valuable tool for automating e-waste sorting, thereby supporting efficient recycling processes and promoting sustainable waste management efforts. Further enhancements could involve advanced hyperparameter tuning or exploring alternative EfficientNetV2 variants for even greater robustness. 

 
