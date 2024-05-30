# Cotton-Plant-Disease-Detection

## Abstract

Cotton is an important crop with a significant role in the global economy, but it is susceptible to various diseases that can significantly impact crop yield and quality. Early detection and diagnosis of these diseases are crucial for efficient crop management and reducing economic losses. In this project, we propose a deep learning-based approach for cotton disease prediction using convolutional neural networks (CNNs). The system is designed to classify six different types of cotton diseases based on their visual symptoms. A large dataset of cotton plant images infected with various diseases was collected and used to train and evaluate the CNN model. The CNN model consists of multiple convolutional layers, pooling layers, and fully connected layers. Transfer learning was applied to fine-tune a pre-trained CNN model for the specific classification task. The proposed system achieved an impressive overall accuracy of 98% on the test dataset, demonstrating its effectiveness in predicting cotton diseases. This system has the potential to serve as a reliable tool for early detection and diagnosis of cotton diseases, thereby improving crop management practices and reducing economic losses.


## Report File

- More documentation
👉 [report file](/Report_File.pdf)


## Data Flow Diagram

<div align="center">
  
![image](https://github.com/kunjankanani/Cotton_Plant_Disease_Detection/assets/115248453/f7df5d67-12c4-4540-a8c2-bb2ea835be5c)

</div>

Let's dive into a detailed explanation of the flowchart :

1. **Image Acquisition**:
   - This initial step involves capturing images, likely of plants or crops, using cameras or other sensors. These images serve as the input for disease prediction.

2. **Image Enhancement**:
   - After acquiring the images, enhancement techniques are applied. These could include adjusting brightness, contrast, and removing noise to improve the quality of the images.
   - Enhanced images are crucial for accurate disease detection.

3. **Image Preprocessing**:
   - In this step, further processing occurs. Common preprocessing steps include resizing, cropping, and normalization.
   - These prepare the images for subsequent analysis by removing irrelevant information and standardizing their format.

4. **Test Annotation**:
   - Annotation involves labeling specific regions of interest within the images. For disease prediction, these annotations might indicate the presence of symptoms or affected areas on the plants.
   - Annotations provide ground truth labels for training machine learning models.

5. **Data Splitting**:
   - The flowchart splits into two paths:
     - **Training Set**: A portion of the annotated images is used for training machine learning models. These models learn patterns from the labeled data.
     - **Test Set**: The remaining images form the test set. They are used to evaluate the model's performance.

6. **Image Augmentation**:
   - The "Augmented Dataset" section suggests that additional data is generated through image augmentation techniques. These techniques create variations of existing images (e.g., rotations, flips, brightness adjustments).
   - Augmentation helps improve model robustness by exposing it to diverse examples.

7. **Model Training**:
   - Using the training set, a machine learning model (such as a neural network) is trained. The model learns to recognize disease patterns based on the annotated images.
   - The "Models" section lists various diseases, including:
     - **Army Worm**
     - **Bacterial Blight**
     - **Cotton Boll Rot**
     - **Green Cotton Boll**
     - **Healthy** (presumably indicating healthy plants)
     - **Powdery Mildew**
     - **Target Spot**
   - Each disease has an associated color-coded box.

8. **Disease Prediction**:
   - The trained model is used to predict whether an unseen image contains a disease or not.
   - If the prediction is positive (i.e., disease present), the flowchart loops back to the beginning of the iterative process.
   - If the prediction is negative (i.e., no disease), it moves forward.

9. **Applying the Model**:
   - The arrow pointing from the "Models" section to the "Test Image" box suggests that this is where the trained model would be applied to make predictions on new, unseen data.

