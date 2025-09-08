from IPython.display import display, Markdown
import pandas as pd

class ResultsManager:
    def __init__(self, y_true, y_pred, y_proba):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.metrics = {}
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate all metrics and store them"""
        from sklearn.metrics import (roc_auc_score, precision_score, recall_score, 
                                    f1_score, average_precision_score, confusion_matrix)
        
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
            'Metric': ['AUC-ROC', 'AUC-PR', 'Precision', 'Recall', 'F1-Score'],
            'Value': [
                f"{self.metrics['auc_roc']:.3f}",
                f"{self.metrics['auc_pr']:.3f}", 
                f"{self.metrics['precision']:.3f}",
                f"{self.metrics['recall']:.3f}",
                f"{self.metrics['f1']:.3f}"
            ]
        })
        
        display(Markdown("### 📊 Model Performance Metrics"))
        display(metrics_df.style.hide(axis="index").set_properties(**{
            'text-align': 'center',
            'font-size': '14px'
        }))
    
    
    def get_markdown_summary(self):
        """Generate markdown text for README"""
        
        return f"""
            ## 📊 Results Summary

            | Metric | Value |
            |--------|-------|
            | **AUC-ROC** | {self.metrics['auc_roc']:.3f} |
            | **AUC-PR** | {self.metrics['auc_pr']:.3f} |
            | **Precision** | {self.metrics['precision']:.3f} |
            | **Recall** | {self.metrics['recall']:.3f} |
            | **F1-Score** | {self.metrics['f1']:.3f} |


            ## 🎯 Key Achievements

            - **Detected {self.metrics['recall']:.1%} of fraudulent transactions** with {self.metrics['precision']:.1%} precision 
            - **Achieved {self.metrics['auc_pr']:.3f} AUC-PR** on extremely imbalanced data ({self.metrics['positive_ratio']:.3%} fraud rate)
            """