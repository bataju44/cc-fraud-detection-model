# Core data handling and computation
import numpy as np
import pandas as pd

# Machine Learning models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import HeUniform, HeNormal, GlorotUniform, GlorotNormal, LecunUniform, LecunNormal
from sklearn.base import BaseEstimator, ClassifierMixin

# Hyperparameter optimization
import optuna

# Model evaluation metrics
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    balanced_accuracy_score,
    accuracy_score
)

# Utilities
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Any, Optional, Union
import os
import json

# Scikit-learn compatibility
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Print versions for reproducibility
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"Optuna Version: {optuna.__version__}")


class BaseOptimizer:
    """
    Base class for all hyperparameter optimizers.
    Handles common functionality and interface.
    """
    
    def __init__(self, X_train, y_train, X_val, y_val, class_weight=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val  # Validation data (X_test before)
        self.y_val = y_val  # Validation labels (y_test before)
        self.class_weight = class_weight
        self.study = None
        self.best_value = None
        self.best_params = None
        
    def objective(self, trial):
        """Objective function to be implemented by child classes"""
        raise NotImplementedError("Child classes must implement this method")
    
    def optimize(self, n_trials=50, direction='maximize'):
        """Run the Optuna optimization study"""
        self.study = optuna.create_study(direction=direction)
        self.study.optimize(self.objective, n_trials=n_trials)
        
        self.best_value = self.study.best_value
        self.best_params = self.study.best_params
        return self.study
    
    def get_best_results(self):
        """Return the best score and parameters"""
        if self.study is None:
            raise ValueError("Optimization has not been run yet. Call .optimize() first.")
        return self.best_value, self.best_params
    
    def get_best_model(self):
        """Return a model initialized with best parameters"""
        raise NotImplementedError("Child classes must implement this method")
    
    def _evaluate_model(self, model):
        """Comprehensive model evaluation with multiple metrics"""
        model.fit(self.X_train, self.y_train)
        
        # Get predictions
        y_pred = model.predict(self.X_val)
        y_pred_proba = self._get_prediction_probabilities(model, self.X_val)
        
        # Calculate multiple metrics
        metrics = {
            'auc_pr': average_precision_score(self.y_val, y_pred_proba),
            'balanced_accuracy': balanced_accuracy_score(self.y_val, y_pred),
            'f1': f1_score(self.y_val, y_pred),
            'precision': precision_score(self.y_val, y_pred),
            'recall': recall_score(self.y_val, y_pred)
        }
        
        # Primary optimization metric (AUC-PR) with secondary checks
        primary_score = metrics['auc_pr']
        
        # Apply penalty if recall is too low (we want to catch fraud!)
        if metrics['recall'] < 0.7:  # At least 70% recall
            primary_score *= 0.7  # 30% penalty
        
        return primary_score
    
    def _get_prediction_probabilities(self, model, X):
        """Safely get prediction probabilities from any model type"""
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)
            return y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(X)
            # Convert to probabilities using sigmoid
            return 1 / (1 + np.exp(-decision_scores))
        else:
            return model.predict(X)
    

class XGBoostOptimizer(BaseOptimizer):
    """Hyperparameter optimizer for XGBoost"""
    
    def objective(self, trial):
        params = {
            'objective': 'binary:logistic',  # Negative Log-Likelihood
            'eval_metric': 'logloss',        # NLL metric
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'scale_pos_weight': self.class_weight,
            'tree_method': 'gpu_hist', # Use GPU acceleration 
            'verbosity': 0,
            'use_label_encoder': False,
            'random_state': 42
        }
        
        model = XGBClassifier(**params)
        return self._evaluate_model(model)
    
    def get_best_model(self):
        best_params = self.best_params.copy()
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'scale_pos_weight': self.class_weight,
            'tree_method': 'gpu_hist',
            'verbosity': 1,
            'random_state': 42
        })
        return XGBClassifier(**best_params)


class LightGBMOptimizer(BaseOptimizer):
    """Hyperparameter optimizer for LightGBM"""
    
    def objective(self, trial):
        params = {
            'objective': 'binary',  # Uses log loss
            'metric': 'binary_logloss',
            'num_leaves': trial.suggest_int('num_leaves', 7, 63),  # Reduced from 20-150
            'max_depth': trial.suggest_int('max_depth', 3, 8),     # Reduced from 3-12
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),  # Increased minimum
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),  # Increased minimum
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),  # Increased minimum
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10, log=True),  # NEW: Add min_child_weight
            'min_split_gain': trial.suggest_float('min_split_gain', 0, 0.1),  # NEW: Add min_split_gain
            'scale_pos_weight': self.class_weight,
            'verbosity': -1,
            'device': 'gpu',
            'random_state': 42,
            'boosting_type': 'gbdt',  # Fixed to gradient boosting
            'n_estimators': 100,      # Fixed number of estimators
        }
        
        model = LGBMClassifier(**params)
        
        try:
            score = self._evaluate_model(model)
            return score
        except Exception as e:
            print(f"LightGBM failed with parameters: {params}")
            print(f"Error: {e}")
            return 0.001  # Return very low score on failure
    
    def get_best_model(self):
        best_params = self.best_params.copy()
        best_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'scale_pos_weight': self.class_weight,
            'verbosity': -1,
            'device': 'gpu',
            'random_state': 42,
            'boosting_type': 'gbdt',
            'n_estimators': 200,  # More estimators for final model
        })
        return LGBMClassifier(**best_params)


class RandomForestOptimizer(BaseOptimizer):
    """Hyperparameter optimizer for Random Forest"""
    
    def objective(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': 'balanced',
            'n_jobs': -1,
            'random_state': 42
        }
        
        model = RandomForestClassifier(**params)
        return self._evaluate_model(model)
    
    def get_best_model(self):
        best_params = self.best_params.copy()
        best_params.update({
            'class_weight': 'balanced',
            'n_jobs': -1,
            'random_state': 42
        })
        return RandomForestClassifier(**best_params)


class SVMOptimizer(BaseOptimizer):
    """Hyperparameter optimizer for SVM"""
    
    def objective(self, trial):
        params = {
            'C': trial.suggest_float('C', 0.1, 10, log=True),
            'gamma': trial.suggest_float('gamma', 0.001, 1, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
            'class_weight': 'balanced',
            'probability': True,
            'random_state': 42
        }
        
        model = SVC(**params)
        return self._evaluate_model(model)
    
    def get_best_model(self):
        best_params = self.best_params.copy()
        best_params.update({
            'class_weight': 'balanced',
            'probability': True,
            'random_state': 42
        })
        return SVC(**best_params)
    

class KerasDNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Advanced DNN classifier with proper weight initializers, batch normalization,
    and comprehensive training monitoring.
    """
    
    def __init__(self, hidden_layers=[64, 32], dropout_rate=0.3, 
                 learning_rate=0.001, activation='relu', 
                 batch_size=32, epochs=50, validation_split=0.2,
                 class_weight=None, verbose=0, 
                 loss='binary_crossentropy',
                 kernel_initializer='he_uniform',
                 use_batch_norm=False,
                 use_skip_connections=False):
        
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.activation = activation
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.class_weight = class_weight
        self.verbose = verbose
        self.loss = loss
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.use_skip_connections = use_skip_connections
        self.model_ = None
        self.history_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.training_history_ = None
        
    def _get_initializer(self):
        """Get the appropriate initializer object"""
        initializer_map = {
            'he_uniform': HeUniform(),
            'he_normal': HeNormal(),
            'glorot_uniform': GlorotUniform(),
            'glorot_normal': GlorotNormal(),
            'lecun_uniform': LecunUniform(),
            'lecun_normal': LecunNormal()
        }
        return initializer_map.get(self.kernel_initializer, HeUniform())
    
    def _validate_initializer_activation(self):
        """Validate that initializer and activation are compatible"""
        relu_initializers = ['he_uniform', 'he_normal']
        tanh_initializers = ['glorot_uniform', 'glorot_normal']
        selu_initializers = ['lecun_uniform', 'lecun_normal']
        
        if self.activation in ['relu', 'leaky_relu', 'elu'] and self.kernel_initializer not in relu_initializers:
            if self.verbose > 0:
                print(f"Warning: {self.kernel_initializer} may be suboptimal for {self.activation}. "
                      f"Consider using he_uniform or he_normal.")
        
        elif self.activation in ['tanh', 'sigmoid'] and self.kernel_initializer not in tanh_initializers:
            if self.verbose > 0:
                print(f"Warning: {self.kernel_initializer} may be suboptimal for {self.activation}. "
                      f"Consider using glorot_uniform or glorot_normal.")
        
        elif self.activation == 'selu' and self.kernel_initializer not in selu_initializers:
            if self.verbose > 0:
                print(f"Warning: {self.kernel_initializer} may be suboptimal for selu. "
                      f"Consider using lecun_uniform or lecun_normal.")
    
    def build_model(self, input_shape):
        """Build advanced DNN model with proper initializers and architecture"""
        self._validate_initializer_activation()
        initializer = self._get_initializer()
        
        model = keras.Sequential()
        model.add(layers.Input(shape=input_shape, name='input_layer'))
        
        # Build hidden layers with advanced architecture
        for i, units in enumerate(self.hidden_layers):
            # Main dense layer
            model.add(layers.Dense(
                units,
                activation=None,  # No activation here for batch norm compatibility
                kernel_initializer=initializer,
                name=f'dense_{i}'
            ))
            
            # Batch normalization before activation
            if self.use_batch_norm:
                model.add(layers.BatchNormalization(name=f'batch_norm_{i}'))
            
            # Activation function
            model.add(layers.Activation(self.activation, name=f'activation_{i}'))
            
            # Dropout
            model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i}'))
        
        # Output layer with appropriate initializer for sigmoid
        model.add(layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer=GlorotUniform(),  # Better for sigmoid output
            name='output_layer'
        ))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss=self.loss,
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='roc_auc'),
                keras.metrics.AUC(name='pr_auc', curve='PR'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.TruePositives(name='tp'),
                keras.metrics.FalsePositives(name='fp'),
                keras.metrics.TrueNegatives(name='tn'),
                keras.metrics.FalseNegatives(name='fn')
            ]
        )
        
        return model
    
    def _create_callbacks(self):
        """Create comprehensive training callbacks"""
        return [
            keras.callbacks.EarlyStopping(
                monitor='val_pr_auc',
                patience=15,
                mode='max',
                restore_best_weights=True,
                min_delta=0.001,
                verbose=self.verbose
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_pr_auc',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                mode='max',
                verbose=self.verbose
            ),
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.CSVLogger(
                '/kaggle/working/dnn_training_log.csv',
                append=True
            ),
            keras.callbacks.ModelCheckpoint(
                '/kaggle/working/best_dnn_model.keras',
                monitor='val_pr_auc',
                save_best_only=True,
                mode='max',
                verbose=self.verbose
            )
        ]
    
    def fit(self, X, y):
        """Fit the DNN model with comprehensive training"""
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels(y)
        
        # Build model
        self.model_ = self.build_model((X.shape[1],))
        
        if self.verbose > 0:
            print("Model Architecture:")
            self.model_.summary()
        
        # Calculate class weights
        if self.class_weight is not None:
            class_weight = {0: 1.0, 1: self.class_weight}
        else:
            class_weight = None
        
        # Train model
        self.history_ = self.model_.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            class_weight=class_weight,
            verbose=self.verbose,
            callbacks=self._create_callbacks(),
            shuffle=True
        )
        
        self.training_history_ = self.history_.history
        return self
    
    def predict(self, X):
        """Predict class labels"""
        check_is_fitted(self)
        X = check_array(X)
        predictions = self.model_.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        check_is_fitted(self)
        X = check_array(X)
        predictions = self.model_.predict(X, verbose=0)
        return np.column_stack([1 - predictions, predictions])
    
    def get_weight_statistics(self):
        """Get comprehensive weight statistics"""
        check_is_fitted(self)
        
        stats = {}
        for i, layer in enumerate(self.model_.layers):
            if hasattr(layer, 'weights') and layer.weights:
                layer_name = layer.name
                weights = [w.numpy() for w in layer.weights]
                
                stats[layer_name] = {
                    'weight_mean': float(weights[0].mean()),
                    'weight_std': float(weights[0].std()),
                    'weight_min': float(weights[0].min()),
                    'weight_max': float(weights[0].max()),
                    'has_bias': len(weights) > 1
                }
                
                if len(weights) > 1:
                    stats[layer_name].update({
                        'bias_mean': float(weights[1].mean()),
                        'bias_std': float(weights[1].std()),
                        'bias_min': float(weights[1].min()),
                        'bias_max': float(weights[1].max())
                    })
        
        return stats
    


class DNNOptimizer(BaseOptimizer):
    """Faster DNN hyperparameter optimizer"""
    
    def objective(self, trial):
        # Suggest kernel initializer
        kernel_initializer = trial.suggest_categorical('kernel_initializer', [
            'he_uniform', 'he_normal'  # Reduced options for speed
        ])
        
        # Suggest activation function
        activation = trial.suggest_categorical('activation', ['relu'])  # Reduced options
        
        # SMALLER NETWORKS for faster training
        params = {
            'hidden_layers': [
                trial.suggest_int(f'layer_{i}_units', 32, 128)  # Reduced from 32-256
                for i in range(trial.suggest_int('n_layers', 2, 3))  # Reduced from 2-5
            ],
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.4),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'activation': activation,
            'kernel_initializer': kernel_initializer,
            'batch_size': trial.suggest_categorical('batch_size', [64, 128]),  # Larger batches
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True]),
            'epochs': 50,  # Reduced from 150 for faster testing
            'class_weight': self.class_weight,
            'verbose': 0,
            'loss': 'binary_crossentropy',
            'validation_split': 0.2
        }
        
        model = KerasDNNClassifier(**params)
        
        try:
            score = self._evaluate_model(model)
            keras.backend.clear_session()
            return score
        except Exception as e:
            keras.backend.clear_session()
            return 0.001
    
    def get_best_model(self):
        best_params = self.best_params.copy()
        best_params.update({
            'class_weight': self.class_weight,
            'epochs': 100,  # More epochs for final training
            'verbose': 1,
            'loss': 'binary_crossentropy',
            'validation_split': 0.2
        })
        
        # Format hidden layers properly
        if 'n_layers' in best_params:
            n_layers = best_params.pop('n_layers')
            hidden_layers = [best_params.pop(f'layer_{i}_units') for i in range(n_layers)]
            best_params['hidden_layers'] = hidden_layers
        
        return KerasDNNClassifier(**best_params)


class ModelComparator:
    """Compare multiple optimized models"""
    
    def __init__(self, X_train, y_train, X_test, y_test, class_weight):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test  # This is TEST data for final evaluation
        self.y_test = y_test  # TEST labels for final evaluation
        self.class_weight = class_weight
        self.models = {}
        self.results = {}
        
    def optimize_and_evaluate(self, model_name, n_trials=30):
        """Optimize and evaluate a specific model with proper validation split"""
        print(f"Optimizing {model_name}...")
        
        # Create validation split from TRAINING data
        from sklearn.model_selection import train_test_split
        
        # Split training data into train/validation for optimization
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
            self.X_train, self.y_train, 
            test_size=0.2, 
            stratify=self.y_train,
            random_state=42
        )
        
        # Select the appropriate optimizer
        optimizers = {
            'xgboost': XGBoostOptimizer,
            'lightgbm': LightGBMOptimizer,
            'random_forest': RandomForestOptimizer,
            'svm': SVMOptimizer,
            'dnn': DNNOptimizer
        }
        
        if model_name not in optimizers:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Use the VALIDATION split for optimization 
        optimizer = optimizers[model_name](
            X_train_opt, y_train_opt,      # Training data for optimization
            X_val_opt, y_val_opt,          # Validation data for optimization
            self.class_weight
        )
        
        # Run optimization
        optimizer.optimize(n_trials=n_trials)
        best_score, best_params = optimizer.get_best_results()
        
        # Train final model on FULL training data
        final_model = optimizer.get_best_model()
        final_model.fit(self.X_train, self.y_train)  # Train on all available data
        
        # Evaluate on TEST data (held-out from the beginning)
        y_pred = final_model.predict(self.X_test)
        y_pred_proba = self._get_prediction_probabilities(final_model, self.X_test)
        
        # Calculate metrics on TEST data
        test_metrics = self._calculate_all_metrics(self.y_test, y_pred, y_pred_proba)
        
        # Store results
        self.models[model_name] = final_model
        self.results[model_name] = {
            'best_score': best_score,        # Validation score from optimization
            'best_params': best_params,      # Best parameters found
            'model': final_model,            # Trained model
            'y_pred': y_pred,                # Test predictions
            'y_pred_proba': y_pred_proba,    # Test probabilities
            'test_metrics': test_metrics     # Test performance metrics
        }
        
        print(f"{model_name} complete. Val AUC-PR: {best_score:.4f}, "
              f"Test AUC-PR: {test_metrics['pr_auc']:.4f}")
        return final_model
    
    def _get_prediction_probabilities(self, model, X):
        """Get prediction probabilities safely"""
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)
            return y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(X)
            return 1 / (1 + np.exp(-decision_scores))
        else:
            return model.predict(X)
    
    def _calculate_all_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive metrics"""
        from sklearn.metrics import confusion_matrix
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba),
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }
    
    def get_comparison_dataframe(self):
        """Create comparison DataFrame with both validation and test metrics"""
        comparison_data = []
        
        for model_name, result in self.results.items():
            test_metrics = result['test_metrics']
            
            metrics = {
                'Model': model_name.upper(),
                'Val_AUC-PR': result['best_score'],  # Validation score
                'Test_AUC-PR': test_metrics['pr_auc'],  # Test score
                'Test_AUC-ROC': test_metrics['roc_auc'],
                'Test_Balanced_Accuracy': test_metrics['balanced_accuracy'],
                'Test_Accuracy': test_metrics['accuracy'],
                'Test_Precision': test_metrics['precision'],
                'Test_Recall': test_metrics['recall'],
                'Test_F1-Score': test_metrics['f1']
            }
            comparison_data.append(metrics)
        
        return pd.DataFrame(comparison_data)