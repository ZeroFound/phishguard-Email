from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.modeling import train_and_save


if __name__ == "__main__":
    bundle = train_and_save()
    print(f"Dataset source: {bundle.metadata['dataset_source']}")
    print(f"Model saved with {bundle.metadata['dataset_rows']} dataset rows.")
    for row in bundle.metadata["model_comparison"]:
        print(
            "{Model}: accuracy={Accuracy:.4f}, precision={Precision:.4f}, "
            "recall={Recall:.4f}, f1={F1-Score:.4f}, roc_auc={ROC-AUC:.4f}, "
            "pr_auc={PR-AUC:.4f}, brier={Brier Score:.4f}".format(**row)
        )
