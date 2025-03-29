import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from collections import Counter
import os
import datetime
import joblib  # Add this import for saving models

# Create directory for saving visualizations
def create_viz_dir():
    """Create directory for saving visualizations"""
    viz_dir = os.path.join(os.getcwd(), "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir

def load_data(file_path):
    """Load the credit card fraud dataset"""
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    return df

def explore_data(df, viz_dir):
    """Perform exploratory data analysis"""
    print("Dataset Head:")
    print(df.head())
    print("\nDataset Description:")
    print(df.describe())
    print("\nClass Distribution:")
    print(df['Class'].value_counts())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Class distribution plot
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df)
    plt.title("Class Distribution")
    plt.savefig(os.path.join(viz_dir, "class_distribution.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation matrix
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
    plt.title("Feature Correlation Matrix")
    plt.savefig(os.path.join(viz_dir, "correlation_matrix.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def feature_engineering(df, viz_dir):
    """Perform feature engineering"""
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df.drop(columns=['Class', 'Time', 'Amount']))
    
    df_viz = df.copy()
    df_viz['PCA1'] = pca_result[:, 0]
    df_viz['PCA2'] = pca_result[:, 1]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Class', data=df_viz)
    plt.title("PCA Visualization")
    plt.savefig(os.path.join(viz_dir, "pca_visualization.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def prepare_data(df, test_size=0.2, random_state=42, apply_smote=True):
    """Prepare data for modeling"""
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    if apply_smote:
        sm = SMOTE(sampling_strategy='auto', random_state=random_state)
        X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
        print("Class distribution after SMOTE:")
        print(Counter(y_train_resampled))
        return X_train_resampled, X_test, y_train_resampled, y_test
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test, viz_dir, model_name):
    """Train and evaluate a model"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(os.path.join(viz_dir, f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    return model

def train_models(X_train, X_test, y_train, y_test, viz_dir):
    """Train and evaluate multiple models"""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        # "SVM": SVC(), # Commented out as it might be slow
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        trained_model = evaluate_model(model, X_train, X_test, y_train, y_test, viz_dir, name)
        results[name] = trained_model
    
    return results

def visualize_feature_importance(model, feature_names, viz_dir):
    """Visualize feature importances for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("This model doesn't have feature_importances_ attribute")

def predict_sample(model, scaler, sample_data):
    """Make a prediction for a sample transaction"""
    sample_data_scaled = scaler.transform(sample_data)
    sample_prediction = model.predict(sample_data_scaled)
    print(f"Sample Transaction Prediction: {'Fraud' if sample_prediction[0] == 1 else 'Non-Fraud'}")
    return sample_prediction

def main():
    
    data_path = os.path.join("data", "creditcard.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: The file {data_path} does not exist.")
        print("Please make sure the creditcard.csv file is in the data folder.")
        return
    
    # Create directory for visualizations
    viz_dir = create_viz_dir()
    print(f"Visualizations will be saved to: {viz_dir}")
    
    # Create directory for model
    model_dir = os.path.join(os.getcwd(), "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Load and explore data
    df = load_data(data_path)
    df = explore_data(df, viz_dir)
    
    # Feature engineering
    df = feature_engineering(df, viz_dir)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Train and evaluate models
    model_results = train_models(X_train, X_test, y_train, y_test, viz_dir)
    
    # Select best model (Random Forest for this example)
    final_model = model_results["Random Forest"]
    
    # Save the model for deployment
    model_path = os.path.join(model_dir, "fraud_detection_model.pkl")
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Scale the data for final model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create and fit a final model
    final_model = RandomForestClassifier(random_state=42)
    final_model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    joblib.dump(final_model, os.path.join(model_dir, 'fraud_detection_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    print(f"Model and scaler saved to: {model_dir}")
    
    # Create sample data with the same feature names as in the training data
    feature_names = X_train.columns.tolist()
    sample_data = pd.DataFrame([
        [0.0,     # Time feature
        -1.2, 1.3, 0.8, -0.5, 1.7, 0.2, 0.6, -0.9, 0.1, 0.4,  # V1-V10 
        -1.0, 0.3, 0.7, -0.2, 1.5, -0.4, 0.1, -0.3, -1.2, 0.6,  # V11-V20
        0.2, 0.7, 0.9, -0.6, -0.3, -0.1, -0.05, -0.02,  # V21-V28
        200.0    # Amount feature
        ]], columns=feature_names)
    
    # Check dimensions to ensure proper input formatting
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Expected feature count: {X_train.shape[1]}")
    
    predict_sample(final_model, scaler, sample_data)
    
    print(f"\nAll visualizations have been saved to: {viz_dir}")

if __name__ == "__main__":
    main()