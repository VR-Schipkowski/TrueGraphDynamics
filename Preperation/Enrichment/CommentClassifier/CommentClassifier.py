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

    
    def classify(self,
                 batch_size: int = 32,
                 temp_dir: Optional[str] = None,
                 steps_to_save =1,
                 ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Klassifiziert alle Kommentare.
        Kann Fortschritt in temp_dir speichern und wieder aufnehmen.

        Rückgabe:
            labels: Liste der vorhergesagten Label
            logits: Liste der Logits pro Kommentar
        """

        total = len(self.comments)
        labels_out: List[str] = [None] * total
        logits_out: List[torch.Tensor] = [None] * total

        # ---------------------------------------------------------
        # 1) Recovery wenn temp_dir existiert
        # ---------------------------------------------------------
        start_index = 0

        if temp_dir is not None:
            os.makedirs(temp_dir, exist_ok=True)
            state_file = os.path.join(temp_dir, "progress.json")

            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state = json.load(f)

                start_index = state.get("next_index", 0)

                # gespeicherte Labels laden
                if "labels" in state:
                    for i, lbl in enumerate(state["labels"]):
                        if lbl is not None:
                            labels_out[i] = lbl

                # gespeicherte Logits laden
                if "logits" in state:
                    for i, logits_list in enumerate(state["logits"]):
                        if logits_list is not None:
                            logits_out[i] = torch.tensor(logits_list)

                print(f"Recovered progress: starting at index {start_index}/{total}")

        # ---------------------------------------------------------
        # 2) Klassifikation ab start_index
        # ---------------------------------------------------------
        for start in tqdm(range(start_index, total, batch_size)):
            end = min(start + batch_size, total)
            batch = self.comments[start:end]

            # Logits berechnen
            batch_logits = self.classify_batch_logits(batch)

            # Labels bestimmen
            for offset, logit in enumerate(batch_logits):
                global_idx = start + offset
                logits_out[global_idx] = logit
                label_idx = int(torch.argmax(logit))
                labels_out[global_idx] = self.labels[label_idx]

            # Speicher sichern
            if temp_dir is not None and steps_to_save is not None and (end % steps_to_save == 0 or end == total) :
                state_file = os.path.join(temp_dir, "progress.json")

                tmp_state = {
                    "next_index": end,
                    "labels": labels_out,
                    "logits": [t.tolist() if isinstance(t, torch.Tensor) else None for t in logits_out]
                }

                with open(state_file, "w") as f:
                    json.dump(tmp_state, f)

                print(f"Saved progress at index {end}/{total}")

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
