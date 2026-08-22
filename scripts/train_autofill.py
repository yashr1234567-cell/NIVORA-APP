#!/usr/bin/env python3
"""
scripts/train_autofill.py
Train a classifier model (DecisionTree vs LogisticRegression) to predict the form 'category'
given 'role' and 'age_bracket'. Saves the model and label encoders to disk.
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def get_age_bracket(age):
    """Maps raw age integer to categorical age brackets."""
    if age < 18:
        return "Under 18"
    elif 18 <= age <= 25:
        return "18-25"
    elif 26 <= age <= 35:
        return "26-35"
    elif 36 <= age <= 50:
        return "36-50"
    else:
        return "50+"

def augment_data_for_missing_features(df, feature_cols, target_col):
    """
    Augment dataset by duplicating rows and masking each feature to -1 one by one.
    This trains the decision tree to handle missing/blank inputs.
    """
    augmented = [df[feature_cols + [target_col]]]
    for col in feature_cols:
        copy_df = df.copy()
        copy_df[col] = -1
        augmented.append(copy_df[feature_cols + [target_col]])
    return pd.concat(augmented, ignore_index=True)

def train_autofill_pipeline():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "form_autofill_mock_data.csv")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)
        
    print(f"Loading mock dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Map ages to age brackets
    print("Mapping ages to age brackets...")
    df['age_bracket'] = df['age'].apply(get_age_bracket)
    
    # 2. Encode categorical columns
    print("Encoding categorical columns...")
    role_encoder = LabelEncoder()
    age_bracket_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    tier_encoder = LabelEncoder()
    
    df['role_encoded'] = role_encoder.fit_transform(df['role'])
    df['age_bracket_encoded'] = age_bracket_encoder.fit_transform(df['age_bracket'])
    df['category_encoded'] = category_encoder.fit_transform(df['category'])
    df['tier_encoded'] = tier_encoder.fit_transform(df['tier'])
    
    # Save encoders to a dictionary for reuse at prediction time
    encoders = {
        'role': role_encoder,
        'age_bracket': age_bracket_encoder,
        'category': category_encoder,
        'tier': tier_encoder
    }
    
    # 3. Augment data to handle missing fields (masked as -1)
    # Features (X): role_encoded, age_bracket_encoded
    # Target (y): category_encoded
    feature_cols = ['role_encoded', 'age_bracket_encoded']
    target_col = 'category_encoded'
    
    df_augmented = augment_data_for_missing_features(df, feature_cols, target_col)
    X = df_augmented[feature_cols]
    y = df_augmented[target_col]
    
    # 4. Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 5. Train and evaluate Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)
    print(f"Decision Tree Classifier Test Accuracy: {dt_acc:.4f}")
    
    # 6. Train and evaluate Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    print(f"Logistic Regression Test Accuracy: {lr_acc:.4f}")
    
    # Compare and choose the best model
    if dt_acc >= lr_acc:
        best_model = dt_model
        best_model_name = "DecisionTreeClassifier"
        best_acc = dt_acc
    else:
        best_model = lr_model
        best_model_name = "LogisticRegression"
        best_acc = lr_acc
        
    print(f"\nSelected '{best_model_name}' as the best model with {best_acc:.4f} accuracy.")
    
    # 7. Save model and encoders to project root
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_pkl_path = os.path.join(workspace_root, "model.pkl")
    encoders_pkl_path = os.path.join(workspace_root, "encoders.pkl")
    
    print(f"Saving model to {model_pkl_path}...")
    joblib.dump(best_model, model_pkl_path)
    
    print(f"Saving encoders to {encoders_pkl_path}...")
    joblib.dump(encoders, encoders_pkl_path)
    
    print("Training pipeline finished successfully!")

if __name__ == "__main__":
    train_autofill_pipeline()
