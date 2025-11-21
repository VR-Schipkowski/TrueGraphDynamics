from ComentClassifier import ComentClassifier
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch


class EmotionCommentClassifierLLM(ComentClassifier):
    """A comment classifier that uses a pre trained emotion detector to classify comments into emotions.
    https://huggingface.co/mrm8488/t5-base-finetuned-emotion"""

    def __init__(self, comments):
        """
        Initialize the EmotionCommentClassifierLLM.

        :param comments: List of comments to classify.
        :param labels: Optional list of labels corresponding to the comments.
        """
        labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        super().__init__(comments, labels)
        self.tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
        self.model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")
        

    def classify_one(self, comment: str) -> str:
        """
        Classify a single comment into an emotion using the language model.

        :param comment: The comment to classify.
        :return: The predicted emotion label.
        """
        return self.classify_batch([comment])[0] 
        

    def classify_batch(self, comments):
        """
        Classify a batch of comments.

        :param comments: List of comments to classify.
        :return: List of predicted emotion labels.
        """
        inputs = self.tokenizer(
            comments,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=4  # Small because labels are short
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [d.strip() for d in decoded]
