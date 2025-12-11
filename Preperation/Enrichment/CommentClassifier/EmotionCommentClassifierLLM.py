import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .CommentClassifier import CommentClassifier
from typing import Sequence
class EmotionCommentClassifierLLM(CommentClassifier):
    def __init__(self, comments: Sequence[str],device=None):
        labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        super().__init__(comments, labels)

        model_name = "j-hartmann/emotion-english-distilroberta-base"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Send to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is not None:
            self.device = device    
        self.model.to(self.device)
        self.model.eval()

    def classify_one(self, comment: str) -> str:
        """Einzelnen Kommentar klassifizieren"""
        logits = self.classify_batch_logits([comment])[0]
        label_idx = int(torch.argmax(logits))
        return self.labels[label_idx]

    def classify_batch_logits(self, comments: Sequence[str]) -> list[torch.Tensor]:
        """
        Batchweise Logits berechnen
        """
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                list(comments),
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # Model Forward
            outputs = self.model(**inputs)
            logits = outputs.logits  # Tensor: [batch_size, num_labels]

        return [logits[i] for i in range(len(comments))]
