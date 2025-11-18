from abc import ABC, abstractmethod
import time
from typing import List, Dict, Optional, Sequence, Tuple
import os
import pandas as pd

class CommentClassifier(ABC):
    # ...existing code...
    def __init__(
        self,
        comments: Sequence [str],
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()  
        self.labels: List[str] = list(labels) if labels is not None else []
        self.comments: List[str] = list(comments)

    def classify(self, batch_size: int = 32, backup_file: Optional[str] = None) -> Tuple[Dict[int, str], pd.DataFrame]:
        """
        Predict labels for all comments and record each step in a DataFrame.
        Returns (mapping, df) where mapping maps comment_id -> label and df
        has columns: comment_id, comment_label, batch_nr, batch_size
        """
        total = len(self.comments)
        cols = ["comment_id", "comment_label", "batch_nr", "batch_size"]

        if backup_file and os.path.exists(backup_file):
            try:
                df = pd.read_csv(backup_file)
                df = df[[c for c in cols if c in df.columns]]
                print(f"Loaded backup from {backup_file}")
            except Exception:
                df = pd.DataFrame(columns=cols)
        else:
            df = pd.DataFrame(columns=cols)

        last_batch_nr = int(df["batch_nr"].max()) if not df.empty else -1
        start_idx = (last_batch_nr + 1) * batch_size

        for batch_nr, start in enumerate(range(start_idx, total, batch_size), start=last_batch_nr + 1):
            t0 = time.perf_counter()
            print(f"Processing batch {batch_nr} / {total // batch_size}")
            end = min(start + batch_size, total)
            batch_comments = self.comments[start:end]
            batch_labels = self.classify_batch(batch_comments)
            batch_df = pd.DataFrame({
                "comment_id": list(range(start, end)),
                "comment_label": batch_labels,
                "batch_nr": batch_nr,
                "batch_size": len(batch_comments),
            })

            df = pd.concat([df, batch_df], ignore_index=True)
            if backup_file:
                df.to_csv(backup_file, index=False)
                print(f"Backup saved to {backup_file}")

            t_elapse = time.perf_counter() - t0
            print(f"Batch classified in {t_elapse:.2f} seconds")
            print(f"Estimated time remaining: {(t_elapse / len(batch_comments)) * (total - end):.2f} seconds")
            print("----------------------------------------------------")
        mapping = {int(r["comment_id"]): r["comment_label"] for _, r in df.iterrows()}
        return mapping, df
         
    @abstractmethod
    def classify_one(self, comment: str) -> str:
        """
        Predict a label for a single comment.
        """
        raise NotImplementedError

    def classify_batch(self, comments: Sequence[str]) -> List[str]:
        """
        Predict labels for a batch of comments using classify_one().
        Subclasses may override for efficiency.
        """
        return [self.classify_one(c) for c in comments]
    # ...existing code...