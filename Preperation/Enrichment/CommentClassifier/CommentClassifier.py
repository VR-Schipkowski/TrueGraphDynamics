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
import csv


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
        output_file: str = "classification_results.csv",
        steps_to_save: int = 1,
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Klassifiziert alle Kommentare, speichert Index/Text/Labels/Logits in CSV,
        und lädt sie am Ende zurück. Recovery möglich durch Auslesen der CSV.
        
        Args:
            batch_size: Größe der Batches für die Klassifizierung
            output_file: Direkter Pfad zur CSV-Datei (z.B. "results/data.csv")
            steps_to_save: Speicherintervall für Progress
        """
        total = len(self.comments)
        start_index = 0
        
        # ----------------- Prepare files -----------------
        # Erstelle Verzeichnis falls nötig
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Bestimme Anzahl der Logits
        num_logits = len(self.labels)
        logit_columns = [f"logit{i+1}" for i in range(num_logits)]
        
        # Recovery: determine start_index from CSV
        if os.path.exists(output_file):
            with open(output_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                last_index = -1
                for row in reader:
                    last_index = int(row["index"])
                start_index = last_index + 1
            print(f"Recovered progress: starting at index {start_index}/{total}")
        else:
            # Erstelle CSV mit Header beim ersten Start
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "text", "label"] + logit_columns)
        
        # ----------------- Classification -----------------
        for start in tqdm(range(start_index, total, batch_size)):
            end = min(start + batch_size, total)
            batch = self.comments[start:end]
            
            batch_logits = self.classify_batch_logits(batch)
            batch_labels = [self.labels[int(torch.argmax(logit))] for logit in batch_logits]
            
            # Append to CSV
            with open(output_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for i, (text, label, logit) in enumerate(zip(batch, batch_labels, batch_logits)):
                    row = [start + i, text, label] + logit.tolist()
                    writer.writerow(row)
            
            # Print progress
            if (end % steps_to_save == 0) or (end == total):
                print(f"Saved progress at index {end}/{total}")
        
        # ----------------- Load all results -----------------
        labels_out = []
        logits_out = []
        
        with open(output_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels_out.append(row["label"])
                # Extrahiere alle logit Spalten
                logits = [float(row[col]) for col in logit_columns]
                logits_out.append(torch.tensor(logits))
        
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
