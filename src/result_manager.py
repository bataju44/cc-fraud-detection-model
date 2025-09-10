from IPython.display import display, Markdown
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                            confusion_matrix, ConfusionMatrixDisplay,
                            roc_auc_score, average_precision_score,
                            precision_score, recall_score, f1_score)

class ResultsManager:
    def __init__(self, y_true, y_pred, y_proba, model_name="Model"):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.model_name = model_name
        self.metrics = {}
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate all metrics and store them"""
        self.metrics = {
            'auc_roc': roc_auc_score(self.y_true, self.y_proba),
            'auc_pr': average_precision_score(self.y_true, self.y_proba),
            'precision': precision_score(self.y_true, self.y_pred),
            'recall': recall_score(self.y_true, self.y_pred),
            'f1': f1_score(self.y_true, self.y_pred),
            'positive_ratio': self.y_true.mean()
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        self.metrics.update({
            'true_negatives': cm[0, 0],
            'false_positives': cm[0, 1],
            'false_negatives': cm[1, 0],
            'true_positives': cm[1, 1]
        })
    
    def show_metrics_table(self):
        """Display metrics as a beautiful table"""
        metrics_df = pd.DataFrame({
            'Metric': ['AUC-ROC', 'AUC-PR', 'Precision', 'Recall', 'F1-Score', 'Fraud Rate'],
            'Value': [
                f"{self.metrics['auc_roc']:.4f}",
                f"{self.metrics['auc_pr']:.4f}", 
                f"{self.metrics['precision']:.4f}",
                f"{self.metrics['recall']:.4f}",
                f"{self.metrics['f1']:.4f}",
                f"{self.metrics['positive_ratio']:.4%}"
            ]
        })
        
        display(Markdown(f"### 📊 {self.model_name} Performance Metrics"))
        display(metrics_df.style.hide(axis="index").set_properties(**{
            'text-align': 'center',
            'font-size': '14px',
            'background-color': '#f8f9fa'
        }))
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC Curve"""
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_name} - ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.show()
    
    def plot_pr_curve(self, save_path=None):
        """Plot Precision-Recall Curve"""
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.3f})')
        
        # Add random baseline
        random_baseline = self.y_true.mean()
        plt.axhline(y=random_baseline, color='red', linestyle='--', 
                   label=f'Random (AUC = {random_baseline:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{self.model_name} - Precision-Recall Curve')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot Confusion Matrix"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                    display_labels=['Genuine', 'Fraud'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'{self.model_name} - Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.show()
    
    def plot_probability_distribution(self, save_path=None):
        """Plot probability distribution for both classes"""
        genuine_probs = self.y_proba[self.y_true == 0]
        fraud_probs = self.y_proba[self.y_true == 1]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(genuine_probs, bins=50, alpha=0.7, color='blue', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Genuine Transactions')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(fraud_probs, bins=50, alpha=0.7, color='red', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Fraudulent Transactions')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.model_name} - Probability Distributions', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curve(self, save_path=None):
        """Plot calibration curve"""
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_true, self.y_proba, n_bins=10, strategy='uniform'
        )
        
        plt.figure(figsize=(10, 8))
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=self.model_name)
        plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'{self.model_name} - Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.show()
    
    def generate_all_plots(self, save_prefix=None):
        """Generate all plots at once"""
        plots = [
            self.plot_roc_curve,
            self.plot_pr_curve,
            self.plot_confusion_matrix,
            self.plot_probability_distribution,
            self.plot_calibration_curve
        ]
        
        plot_names = ['roc_curve', 'pr_curve', 'confusion_matrix', 'probability_dist', 'calibration']
        
        for plot_func, name in zip(plots, plot_names):
            save_path = f"{save_prefix}_{name}.png" if save_prefix else None
            plot_func(save_path)
    
    def get_markdown_summary(self):
        """Generate markdown text for README"""
        return f"""
## 📊 {self.model_name} Results Summary

| Metric | Value |
|--------|-------|
| **AUC-ROC** | {self.metrics['auc_roc']:.4f} |
| **AUC-PR** | {self.metrics['auc_pr']:.4f} |
| **Precision** | {self.metrics['precision']:.4f} |
| **Recall** | {self.metrics['recall']:.4f} |
| **F1-Score** | {self.metrics['f1']:.4f} |
| **Fraud Detection Rate** | {self.metrics['recall']:.2%} |

## 🎯 Key Achievements

- **Detected {self.metrics['recall']:.1%} of fraudulent transactions** with {self.metrics['precision']:.1%} precision
- **Achieved {self.metrics['auc_pr']:.4f} AUC-PR** on extremely imbalanced data ({self.metrics['positive_ratio']:.4%} fraud rate)
- **Overall performance score**: {self.metrics['f1']:.4f} F1-Score
"""
    
    def print_detailed_report(self):
        """Print comprehensive performance report"""
        display(Markdown(f"# 🎯 {self.model_name} - Detailed Performance Report"))
        self.show_metrics_table()
        
        display(Markdown("## 📈 Performance Visualizations"))
        self.generate_all_plots()
        
        display(Markdown("## 📋 Confusion Matrix Details"))
        cm_data = pd.DataFrame({
            '': ['Actual Genuine', 'Actual Fraud'],
            'Predicted Genuine': [
                self.metrics['true_negatives'], 
                self.metrics['false_negatives']
            ],
            'Predicted Fraud': [
                self.metrics['false_positives'],
                self.metrics['true_positives']
            ]
        })
        display(cm_data.style.hide(axis="index"))