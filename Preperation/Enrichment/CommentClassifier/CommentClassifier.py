from abc import ABC, abstractmethod
import json
import os
from typing import List, Optional, Sequence, Tuple
import torch
from tqdm.notebook import tqdm


class CommentClassifier(ABC):
    def __init__(
        self,
        comments: Sequence[str],
        labels: Sequence[str] = None,
    ) -> None:
        super().__init__()
        self.labels: List[str] = list(labels) if labels else []
        self.comments: List[str] = list(comments)

    from tqdm.notebook import tqdm
import os
import json
import torch
from typing import List, Tuple, Sequence
from abc import ABC, abstractmethod

class CommentClassifier(ABC):
    def __init__(
        self,
        comments: Sequence[str],
        labels: Sequence[str] = None,
    ) -> None:
        super().__init__()
        self.labels: List[str] = list(labels) if labels else []
        self.comments: List[str] = list(comments)

    @abstractmethod
    def classify_batch_logits(self, batch: List[str]) -> List[torch.Tensor]:
        """Must be implemented by subclass."""
        pass

    def classify(
        self,
        batch_size: int = 32,
        temp_dir: str = "temp_classification",
        steps_to_save: int = 1,
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Klassifiziert alle Kommentare, speichert Labels/Logits direkt auf der Festplatte,
        und lädt sie am Ende zurück. Recovery möglich. temp_dir ist jetzt verpflichtend.
        """
        total = len(self.comments)
        start_index = 0

        # ----------------- Prepare temp_dir -----------------
        os.makedirs(temp_dir, exist_ok=True)
        state_file = os.path.join(temp_dir, "progress.json")
        labels_file = os.path.join(temp_dir, "labels.jsonl")
        logits_file = os.path.join(temp_dir, "logits.jsonl")

        # Recovery: determine start_index
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
                start_index = state.get("next_index", 0)
            print(f"Recovered progress: starting at index {start_index}/{total}")

        # ----------------- Classification -----------------
        for start in tqdm(range(start_index, total, batch_size)):
            end = min(start + batch_size, total)
            batch = self.comments[start:end]

            batch_logits = self.classify_batch_logits(batch)
            batch_labels = [self.labels[int(torch.argmax(logit))] for logit in batch_logits]

            # Append labels
            with open(labels_file, "a") as f:
                for label in batch_labels:
                    f.write(json.dumps(label) + "\n")

            # Append logits
            with open(logits_file, "a") as f:
                for logit in batch_logits:
                    f.write(json.dumps(logit.tolist()) + "\n")

            # Save progress
            if (end % steps_to_save == 0) or (end == total):
                with open(state_file, "w") as f:
                    json.dump({"next_index": end}, f)
                print(f"Saved progress at index {end}/{total}")

        # ----------------- Load all results -----------------
        labels_out = []
        with open(labels_file, "r") as f:
            for line in f:
                labels_out.append(json.loads(line))

        logits_out = []
        with open(logits_file, "r") as f:
            for line in f:
                logits_out.append(torch.tensor(json.loads(line)))

        return labels_out, logits_out

    @abstractmethod
    def classify_one(self, comment: str) -> str:
        raise NotImplementedError

    def classify_batch(self, comments: Sequence[str]) -> List[str]:
        """Standard-Implementierung über classify_one()"""
        return [self.classify_one(c) for c in comments]

    @abstractmethod
    def classify_batch_logits(self, comments: Sequence[str]) -> List[torch.Tensor]:
        """
        Predict logits for a batch of comments.
        Muss von Subklasse implementiert werden.
        """
        raise NotImplementedError
