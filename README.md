# Earthquake-Prediction-with-Hill-Climbing-Hyperparameter-Tuning

## Overview
This repository presents a **project** on optimizing neural network models for forecasting earthquakes using **hill-climbing hyperparameter tuning**.  
The project demonstrates how a **lightweight search algorithm** can iteratively improve model architecture and training settings without requiring complex frameworks.  

Unlike grid search or random search, hill-climbing evaluates local “neighbor” configurations and adopts better candidates step-by-step, making it efficient and interpretable.

---

## Objectives
- Implement a **feed-forward neural network** for forecasting.  
- Explore **hill-climbing search** as a reproducible approach for hyperparameter tuning.  
- Tune model hyperparameters including:  
  - Number of neurons per hidden layer  
  - Batch size  
  - Learning rate  
- Evaluate model performance with **validation loss** and **mean absolute error (MAE)**.    

---

## Methodology
1. **Data Preparation**  
   - Train/test split with a selected subset of features (`best_features`).  
   - Inputs standardized for stable optimization.  

2. **Model Architecture**  
   - Built with **TensorFlow/Keras**.  
   - Input: selected feature set.  
   - Hidden layers: customizable list of neurons (e.g., `[32, 16]`).  
   - Output: regression prediction with `mse` loss and `mae` metric.  

3. **Search Space**  
   - Hidden layer neurons: `[8, 16, 32, 64]`  
   - Batch sizes: `[16, 32, 64]`  
   - Learning rates: `[0.001, 0.0005, 0.0001]`  

4. **Hill-Climbing Algorithm**  
   - Start from **user-defined configuration**.  
   - Generate candidate neighbors by:  
     - Replacing neurons in each layer  
     - Trying alternative batch sizes  
     - Trying alternative learning rates  
   - Evaluate each candidate with early stopping (`val_loss`).  
   - Adopt the first improvement and repeat until no better neighbor is found.  

5. **Training Strategy**  
   - Early stopping (`patience=20`) to avoid overfitting.  
   - Validation split (`0.2`) for consistent comparisons.  
   - Performance measured with `val_loss` and `mae`.  


## Key Results
- Hill-climbing identified improved configurations over the initial user settings.  
- Demonstrated sensitivity of performance to **layer sizes, batch size, and learning rate**.  
- Validated that even simple search heuristics can provide **systematic improvements** without exhaustive grid search.
