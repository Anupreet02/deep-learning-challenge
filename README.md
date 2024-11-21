# deep-learning-challenge

* [ ] **Overview**

The objective of this analysis is to build a binary classification model using deep learning to predict whether a charity funded by Alphabet Soup will be successful. This model will enable stakeholders to make informed decisions about allocating resources to charities likely to succeed. The analysis involves multiple steps, including data preprocessing, model training and evaluation, optimization, and reporting the results.

* [ ] **Purpose of the Analysis**

The purpose of this analysis is to classify charities as successful or not, using data provided by Alphabet Soup. By developing a predictive model, Alphabet Soup can streamline funding decisions and focus efforts on organizations most likely to achieve their goals. The analysis aims to optimize the model for accuracy while documenting the process and findings comprehensively.

* [ ] **Results**

1. Data Preprocessing

* Target Variable: The target variable is IS_SUCCESSFUL, indicating whether a charity achieved its intended success.
* Feature Variables: Features include application type, income amount, category codes, organization type, use cases, and special considerations.
* Dropped Variables: The columns EIN and NAME were removed as they are identifiers with no predictive value.
* Handling Categorical Data: Categorical variables with more than 10 unique values were grouped into an "Other" category. and One-hot encoding was applied to categorical features to convert them into numeric format.
* Scaling the Features: StandardScaler was used to normalize numerical data to ensure consistency and improve model convergence.
* Data Splitting: The dataset was split into training (80%) and testing (20%) sets using train_test_split.

2. **Model Architecture**

* Neurons and Layers: The initial model was designed with: **Input Layer:** Accepting the preprocessed features. **Hidden Layers**: Layer 1: 128 neurons with ReLU activation, Layer 2: 64 neurons with ReLU activation**. Output Layer**: A single neuron with a Sigmoid activation function for binary classification.
* Compilation: **Optimizer**: Adam (adaptive learning rate for faster convergence), **Loss Function**: Binary Cross-Entropy (suitable for binary classification tasks), **Evaluation Metric**: Accuracy.
* Training Configuration: The model was trained for 100 epochs with a 20% validation split, using EarlyStopping to avoid overfitting.

3. **Model Performance**

* Baseline Performance: Accuracy : 73% accuracy
* Optimized Performance: Adjustments such as increasing hidden layers, neurons, and experimenting with activation functions improved accuracy to 78.57%

* [ ] **Optimization Steps**

To achieve a target accuracy of 75%, the following optimizations were attempted:

1. **Data Adjustments** :

* Encoding categorical variables with `pd.get_dummies()`.
* Combining rare categorical values into "Other".
* Scaling features using `StandardScaler()`.

1. **Model Adjustments** :

* Increased the number of neurons in hidden layers.
* Added more hidden layers for greater learning capacity.
* Experimented with different activation functions (`relu`, `tanh`).

1. **Training Adjustments** :

* Increased the number of epochs.
* Used EarlyStopping and ModelCheckpoint callbacks to improve training efficiency.

* [ ] **Final Model**

* **Accuracy** : Despite extensive optimization, the highest accuracy achieved was ~73%. Then I had to try different approach that is focused on identifying and handling low-frequency categories in a dataset:
* *Replacing with "Other"* : Consolidate rare categories into a single "Other" category.
* *Dropping* : Remove the data associated with these rare categories (only if justified).
* *Encoding Separately* : Treat them differently in downstream processing.

* [ ] **Analysis of Results**

***Answers to Specific Questions***

1. Which features were used as inputs?

Categorical features such as application type, income amount, organization type, and special considerations, converted to numerical format using one-hot encoding, were used.

2. What was the target variable?

The target variable was IS_SUCCESSFUL, indicating whether a charity achieved success.

3. What methods were used to preprocess the data?

Dropped irrelevant columns, grouped rare categories, applied one-hot encoding to categorical variables, scaled numeric features using StandardScaler, and split the data into training and testing sets.

4. How was the model structured?

A feedforward neural network with two hidden layers (128 and 64 neurons), ReLU activation for hidden layers, and a Sigmoid activation for the output layer.

5. What were the model’s performance metrics?

Accuracy: 73.0% (optimized model) and Loss: Reduced from 0.577 to a slightly lower value during optimization attempts.

6. How was the model optimized?

* The model was optimized by:
* Experimenting with the number of layers and neurons.
* Adjusting the learning rate and epochs.
* Using early stopping to avoid overfitting.

* [ ] **Comparison of Training and Validation Accuracy and Loss**

During training, the training accuracy steadily increased, while validation accuracy was slightly lower, indicating some overfitting. The training loss decreased over time, but the validation loss occasionally plateaued or increased, suggesting the model struggled to generalize to unseen data. These trends highlight the importance of monitoring both training and validation metrics to ensure the model does not overfit and performs well on new data.

* [ ] **Summary of Results**

The final deep learning model successfully classified charities with an accuracy of  **78.57%** , exceeding the baseline performance. Data preprocessing steps, model architecture refinements, and iterative optimizations contributed to these results. Despite significant improvements, achieving an accuracy closer to 80% or beyond remains a future goal.

**Conclusion** : While the deep learning model achieved notable accuracy, experimenting with a Random Forest Classifier could provide additional insights and potentially higher accuracy. Combining the strengths of both models could create a more robust decision-support system for Alphabet Soup.
