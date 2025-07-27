# train_and_save_model.py
# Fixed Model Training Script - English Version

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_realistic_heart_data():
    """Create realistic heart disease dataset with proper correlations"""
    np.random.seed(42)
    n_samples = 1000
    
    print("Creating realistic heart disease dataset...")
    
    # Generate correlated features for more realistic data
    data = {}
    
    # Age (higher age = higher risk)
    data['age'] = np.random.normal(52, 10, n_samples).astype(int)
    data['age'] = np.clip(data['age'], 29, 77)
    
    # Sex (1=male, 0=female) - males have higher risk
    data['sex'] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    
    # Chest pain type (0-3) - correlated with heart disease
    data['cp'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.25, 0.25, 0.2])
    
    # Resting blood pressure - correlated with age
    base_bp = 120 + (data['age'] - 50) * 0.8
    data['trestbps'] = np.random.normal(base_bp, 15).astype(int)
    data['trestbps'] = np.clip(data['trestbps'], 94, 200)
    
    # Cholesterol - correlated with age and sex
    base_chol = 200 + (data['age'] - 50) * 2 + data['sex'] * 20
    data['chol'] = np.random.normal(base_chol, 40).astype(int)
    data['chol'] = np.clip(data['chol'], 126, 564)
    
    # Fasting blood sugar > 120 mg/dl
    fbs_prob = 0.1 + (data['age'] > 55) * 0.1
    data['fbs'] = np.random.binomial(1, fbs_prob, n_samples)
    
    # Resting ECG results
    data['restecg'] = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
    
    # Maximum heart rate - inversely correlated with age
    base_thalach = 220 - data['age'] + np.random.normal(0, 20, n_samples)
    data['thalach'] = base_thalach.astype(int)
    data['thalach'] = np.clip(data['thalach'], 71, 202)
    
    # Exercise induced angina
    exang_prob = 0.2 + (data['age'] > 55) * 0.2
    data['exang'] = np.random.binomial(1, exang_prob, n_samples)
    
    # ST depression
    data['oldpeak'] = np.random.exponential(1.2, n_samples)
    data['oldpeak'] = np.clip(data['oldpeak'], 0, 6.2)
    
    # ST segment slope
    data['slope'] = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])
    
    # Number of major vessels (0-3)
    ca_prob = np.clip((data['age'] - 40) / 40, 0, 1) * 0.3
    data['ca'] = np.random.poisson(ca_prob * 2, n_samples)
    data['ca'] = np.clip(data['ca'], 0, 3)
    
    # Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)
    data['thal'] = np.random.choice([1, 2, 3], n_samples, p=[0.7, 0.15, 0.15])
    
    # Create realistic target variable based on multiple risk factors
    targets = []
    for i in range(n_samples):
        risk_score = 0
        
        # Age factor (stronger for older patients)
        if data['age'][i] > 65: risk_score += 3
        elif data['age'][i] > 55: risk_score += 2
        elif data['age'][i] > 45: risk_score += 1
        
        # Gender factor (males higher risk)
        if data['sex'][i] == 1: risk_score += 1.5
        
        # Chest pain factor (asymptomatic is concerning)
        if data['cp'][i] == 3: risk_score += 2
        elif data['cp'][i] == 0: risk_score += 1
        
        # Blood pressure factor
        if data['trestbps'][i] > 160: risk_score += 2
        elif data['trestbps'][i] > 140: risk_score += 1
        
        # Cholesterol factor
        if data['chol'][i] > 280: risk_score += 2
        elif data['chol'][i] > 240: risk_score += 1
        
        # Heart rate factor (low max heart rate is concerning)
        if data['thalach'][i] < 100: risk_score += 2
        elif data['thalach'][i] < 120: risk_score += 1
        
        # Exercise angina factor
        if data['exang'][i] == 1: risk_score += 1.5
        
        # ST depression factor
        if data['oldpeak'][i] > 3.0: risk_score += 2
        elif data['oldpeak'][i] > 2.0: risk_score += 1
        
        # Major vessels factor
        if data['ca'][i] >= 2: risk_score += 2
        elif data['ca'][i] >= 1: risk_score += 1
        
        # Thalassemia factor
        if data['thal'][i] == 3: risk_score += 1.5  # Reversible defect
        elif data['thal'][i] == 2: risk_score += 1   # Fixed defect
        
        # ECG factor
        if data['restecg'][i] == 2: risk_score += 1  # LV hypertrophy
        elif data['restecg'][i] == 1: risk_score += 0.5  # ST-T abnormality
        
        # Fasting blood sugar
        if data['fbs'][i] == 1: risk_score += 0.5
        
        # Add some controlled randomness
        risk_score += np.random.uniform(-1, 1)
        
        # Convert to binary target (threshold around 4-5)
        target = 1 if risk_score >= 4.5 else 0
        targets.append(target)
    
    data['target'] = targets
    df = pd.DataFrame(data)
    
    # Ensure reasonable distribution
    positive_ratio = df['target'].mean()
    print(f"Positive cases ratio: {positive_ratio:.3f}")
    
    # If ratio is too skewed, adjust
    if positive_ratio < 0.3 or positive_ratio > 0.7:
        print("Adjusting target distribution for better balance...")
        # Randomly flip some labels to balance
        n_flip = int(abs(0.5 - positive_ratio) * len(df) * 0.5)
        if positive_ratio < 0.5:
            # Need more positive cases
            neg_indices = df[df['target'] == 0].index
            flip_indices = np.random.choice(neg_indices, min(n_flip, len(neg_indices)), replace=False)
            df.loc[flip_indices, 'target'] = 1
        else:
            # Need more negative cases
            pos_indices = df[df['target'] == 1].index
            flip_indices = np.random.choice(pos_indices, min(n_flip, len(pos_indices)), replace=False)
            df.loc[flip_indices, 'target'] = 0
    
    final_ratio = df['target'].mean()
    print(f"Final positive cases ratio: {final_ratio:.3f}")
    
    return df

def train_and_save_model():
    """Train the model and save all required files"""
    
    print("=" * 60)
    print("HEART DISEASE PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Create dataset
    df = create_realistic_heart_data()
    
    print(f"\nDataset created successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:")
    print(df['target'].value_counts())
    print(f"No Disease: {(df['target'] == 0).sum()} ({(df['target'] == 0).mean():.1%})")
    print(f"Heart Disease: {(df['target'] == 1).sum()} ({(df['target'] == 1).mean():.1%})")
    
    # Display feature statistics
    print(f"\nFeature Statistics:")
    print(df.describe())
    
    # Separate features and target
    feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    X = df[feature_columns]
    y = df['target']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Feature scaling
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled successfully!")
    
    # Train Random Forest model
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,          # More trees for better performance
        max_depth=15,              # Deeper trees
        min_samples_split=5,       # Minimum samples to split
        min_samples_leaf=2,        # Minimum samples in leaf
        max_features='sqrt',       # Number of features for best split
        bootstrap=True,            # Bootstrap sampling
        oob_score=True,           # Out-of-bag score
        random_state=42,          # For reproducibility
        n_jobs=-1                 # Use all available cores
    )
    
    model.fit(X_train_scaled, y_train)
    print("Model training completed!")
    
    # Model evaluation
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Training predictions
    y_train_pred = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Test predictions
    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Out-of-bag score
    oob_score = model.oob_score_
    
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy:.1%})")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.1%})")
    print(f"Out-of-Bag Score: {oob_score:.4f} ({oob_score:.1%})")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['No Disease', 'Heart Disease']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"Actual    No Disease  Heart Disease")
    print(f"No Disease      {cm[0,0]:3d}         {cm[0,1]:3d}")
    print(f"Heart Disease   {cm[1,0]:3d}         {cm[1,1]:3d}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save model files
    print("\n" + "="*50)
    print("SAVING MODEL FILES")
    print("="*50)
    
    try:
        # Save main model
        joblib.dump(model, 'heart_disease_model.pkl')
        print("✅ heart_disease_model.pkl saved successfully!")
        
        # Save scaler
        joblib.dump(scaler, 'scaler.pkl')
        print("✅ scaler.pkl saved successfully!")
        
        # Save feature columns for reference
        joblib.dump(feature_columns, 'feature_columns.pkl')
        print("✅ feature_columns.pkl saved successfully!")
        
        # Save a sample of training data for SHAP analysis (if needed later)
        sample_size = min(100, len(X_train_scaled))
        X_train_sample = X_train_scaled[:sample_size]
        joblib.dump(X_train_sample, 'X_train_scaled_sample.pkl')
        print("✅ X_train_scaled_sample.pkl saved successfully!")
        
        # Save feature importance
        feature_importance.to_csv('feature_importance.csv', index=False)
        print("✅ feature_importance.csv saved successfully!")
        
        # Save model metadata
        metadata = {
            'model_type': 'RandomForestClassifier',
            'n_estimators': 200,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'oob_score': oob_score,
            'feature_columns': feature_columns,
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_size': len(df),
            'positive_ratio': df['target'].mean()
        }
        joblib.dump(metadata, 'model_metadata.pkl')
        print("✅ model_metadata.pkl saved successfully!")
        
        print(f"\n🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📈 Final Test Accuracy: {test_accuracy:.1%}")
        print(f"📁 All model files saved in current directory")
        
        # List all created files
        print(f"\n📋 Files Created:")
        files_created = [
            'heart_disease_model.pkl',
            'scaler.pkl', 
            'feature_columns.pkl',
            'X_train_scaled_sample.pkl',
            'feature_importance.csv',
            'model_metadata.pkl'
        ]
        
        for i, file in enumerate(files_created, 1):
            print(f"   {i}. {file}")
        
        return model, scaler, test_accuracy, feature_importance
        
    except Exception as e:
        print(f"❌ Error saving model files: {str(e)}")
        return None, None, None, None

def test_saved_model():
    """Test the saved model to ensure it works correctly"""
    print("\n" + "="*50)
    print("TESTING SAVED MODEL")
    print("="*50)
    
    try:
        # Load saved model and scaler
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        
        print("✅ Model files loaded successfully!")
        
        # Create a test sample
        test_sample = pd.DataFrame({
            'age': [45], 'sex': [1], 'cp': [2], 'trestbps': [130], 
            'chol': [250], 'fbs': [0], 'restecg': [0], 'thalach': [150], 
            'exang': [0], 'oldpeak': [1.5], 'slope': [1], 'ca': [0], 'thal': [2]
        })
        
        # Scale and predict
        test_sample_scaled = scaler.transform(test_sample)
        prediction = model.predict(test_sample_scaled)
        probability = model.predict_proba(test_sample_scaled)
        
        print(f"Test sample prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
        print(f"Prediction probabilities: No Disease: {probability[0][0]:.3f}, Heart Disease: {probability[0][1]:.3f}")
        
        print("✅ Model testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing saved model: {str(e)}")

if __name__ == "__main__":
    print("Starting Heart Disease Prediction Model Training...")
    
    # Train and save model
    model, scaler, accuracy, feature_importance = train_and_save_model()
    
    if model is not None:
        # Test the saved model
        test_saved_model()
        
        print(f"\n🏆 TRAINING SUMMARY:")
        print(f"   • Model: Random Forest Classifier")
        print(f"   • Accuracy: {accuracy:.1%}")
        print(f"   • Status: Ready for deployment!")
        print(f"\n💡 You can now run your Streamlit app with the trained model!")
    else:
        print(f"\n❌ Training failed. Please check the error messages above.")