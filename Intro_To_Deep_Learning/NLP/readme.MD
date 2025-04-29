# Disaster Tweet Classification

This project implements a machine learning solution to classify tweets as either relating to real disasters or not. Using natural language processing techniques and a BiLSTM neural network architecture, the model can distinguish between tweets about actual disasters and those using disaster-related vocabulary in non-disaster contexts.

## Problem Description

Social media platforms like Twitter are increasingly used during emergency situations to communicate and share information. However, not all tweets containing disaster-related keywords actually report real disasters. This project addresses the challenge of automatically identifying whether a tweet is announcing a real disaster (requiring attention from emergency services) or using disaster-related language in a non-emergency context.

**Dataset:**
- Source: Real World Disaster Tweets dataset from Kaggle
- Size: Approximately 7,600 tweets (57% non-disaster, 43% disaster)
- Structure: Tweet text, keywords, locations, and binary label (1=disaster, 0=non-disaster)
- Features: Text content, plus engineered features including text length, word count, and presence of URLs, mentions, and hashtags

## Exploratory Data Analysis

The EDA revealed several important patterns in the data:

1. **Class Distribution**: The dataset has a slight imbalance with ~57% non-disaster tweets and ~43% disaster tweets.

2. **Text Length**: Disaster tweets tend to be slightly longer (median ~115 characters) than non-disaster tweets (median ~100 characters).

3. **Word Count**: Both classes have similar word counts, with most tweets containing 5-25 words.

4. **Special Features Analysis**:
   - Tweets containing URLs are more likely to be disaster-related (55%)
   - Tweets with mentions tend to be non-disaster tweets (68%)
   - Hashtag presence shows minimal predictive power

5. **Keyword Analysis**: Many disaster-related terms appear in both classes, but with different frequencies and contexts:
   - Terms like "outbreak," "wreckage," and "derailment" strongly indicate actual disasters
   - Terms like "armageddon," "fear," and "obliteration" often appear in non-disaster contexts

6. **Location Analysis**: Geographic variation exists, with tweets from India and Nigeria having the highest disaster rates (>80%), while urban US locations like New York show lower rates (~22%).

7. **Data Cleaning**: Preprocessing steps included:
   - Removing URLs, mentions, and special characters
   - Lowercasing and tokenizing text
   - Handling missing values in text fields
   - Creating engineered features for model input

## Model Architecture

Based on the EDA and the nature of the classification task, I implemented a Bidirectional LSTM (BiLSTM) neural network with the following architecture:

```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 150, 100)          2000000   
                                                                 
 batch_normalization (Batch  (None, 150, 100)          400       
 Normalization)                                                  
                                                                 
 bidirectional (Bidirection  (None, 150, 128)          84480     
 al)                                                             
                                                                 
 batch_normalization_1 (Bat  (None, 150, 128)          512       
 chNormalization)                                                
                                                                 
 bidirectional_1 (Bidirecti  (None, 150, 64)           41216     
 onal)                                                           
                                                                 
 global_max_pooling1d (Glob  (None, 64)                0         
 alMaxPooling1D)                                                 
                                                                 
 dense (Dense)               (None, 64)                4160      
                                                                 
 dropout (Dropout)           (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 64)                4160      
                                                                 
 dropout_1 (Dropout)         (None, 64)                0         
                                                                 
 dense_2 (Dense)             (None, 1)                 65        
                                                                 
=================================================================
Total params: 2,134,993
Trainable params: 2,134,537
Non-trainable params: 456
_________________________________________________________________
```

**Key Components:**

1. **Word Embeddings**: Embedding layer with L2 regularization (1e-5) and zero masking for handling variable-length sequences.

2. **Stacked Bidirectional LSTM Layers**: Two BiLSTM layers (64 and 32 units) to capture complex sequential patterns, with both regular dropout and recurrent dropout (0.3) to prevent overfitting.

3. **Batch Normalization**: Applied after embedding and first BiLSTM layer to stabilize training and accelerate convergence.

4. **Global Max Pooling**: Extracts the most important features from the LSTM outputs, significantly reducing dimensionality.

5. **Dense Layers**: Two regularized dense layers (64 units each) with ReLU activation and L2 regularization (1e-4), separated by dropout (0.3).

6. **Output Layer**: Single sigmoid node for binary classification with threshold at 0.5.

**Implementation Details:**
- Used Adam optimizer with learning rate of 0.001
- Tracked multiple metrics beyond accuracy: precision, recall, and AUC
- Applied comprehensive regularization strategy (L2, dropout, batch normalization)
- Used binary cross-entropy loss function
- Implemented early stopping and class weights to address the slight class imbalance

## Results and Analysis

The BiLSTM model achieved strong performance on the test data:

- **Accuracy**: ~80%
- **ROC-AUC**: 0.84
- **F1-Score**: 0.75
- **Confusion Matrix**:
  - True Negatives: 773
  - False Positives: 96
  - False Negatives: 204
  - True Positives: 450

**Hyperparameter Tuning Results:**
Various configurations were tested, with the following parameters yielding optimal results:
- Embedding dimension: 100
- LSTM units: 64 (bidirectional = 128 total)
- Batch size: 32
- Learning rate: 0.001
- Dropout rate: 0.3

**Performance Analysis:**
- The model shows higher precision (82%) than recall (69%) for disaster tweets
- Training curves indicate potential overfitting after epoch 2
- URL presence emerged as the strongest numerical feature
- The model successfully distinguishes between literal and figurative uses of disaster vocabulary
- Bimodal probability distribution indicates high confidence in many predictions

**Error Analysis:**
- Most false negatives (missed disasters) lack URLs or specific disaster terms
- Some false positives contain disaster terms used in abstract or hypothetical contexts
- Geographical bias exists, with better performance on tweets from certain regions

## Conclusion and Future Work

The BiLSTM model successfully tackles the challenge of disaster tweet classification, achieving 80% accuracy and demonstrating strong discrimination between literal and figurative uses of disaster-related language. The model effectively leverages both textual content and metadata features, with URL presence and text length providing particularly valuable signals.

**Key Learnings:**
- Context is crucial for interpreting disaster-related terms
- BiLSTM architecture effectively captures sequential relationships in text
- Engineered features enhance performance beyond text-only approaches
- Class imbalance handling improves overall model performance

**Future Improvements:**
1. **Threshold Optimization**: Experiment with different classification thresholds to find optimal precision-recall balance
2. **Model Ensemble**: Combine BiLSTM with other architectures like BERT for improved performance
3. **Advanced Feature Engineering**: Incorporate sentiment analysis and named entity recognition
4. **Data Augmentation**: Generate synthetic disaster tweets to address class imbalance
5. **Attention Mechanisms**: Implement to help the model focus on critical parts of tweets
6. **Error Analysis**: Conduct deeper analysis of misclassified examples