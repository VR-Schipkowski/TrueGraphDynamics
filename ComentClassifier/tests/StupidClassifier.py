from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ComentClassifier import CommentClassifier
import pandas as pd
import os

class StupidClassifier(CommentClassifier):
    """A simple classifier that labels comments as 'positive' if they contain the word 'good', else 'negative'.
    It is used to test the CommentClassifier base class functionality."""
    def __init__(self, comments, labels=None):
        super().__init__(comments, labels)

    def classify_one(self, comment: str) -> str:
        return "positive" if "good" in comment.lower() else "negative"

    def classify_batch(self, comments):
        return [self.classify_one(c) for c in comments]

if __name__ == "__main__":
    comments = ["This is good", "Bad experience", "Good job!", "neutral"]
    backup = "backup_test.csv"

    # ensure clean start
    try:
        os.remove(backup)
    except OSError:
        pass

    # first run: produce backup file via base classify
    clf = StupidClassifier(comments)
    mapping1, df1 = clf.classify(batch_size=2, backup_file=backup)
    print("first mapping:", mapping1)
    print(df1)

    # second run: should resume / read from backup and not reclassify already processed items
    clf2 = StupidClassifier(comments)
    mapping2, df2 = clf2.classify(batch_size=2, backup_file=backup)
    print("second mapping:", mapping2)
    print(df2)

    # basic assertions
    assert mapping1 == mapping2
    df_file = pd.read_csv(backup)
    assert df_file.shape[0] == df1.shape[0]
    assert set(mapping1.keys()) == set(range(len(comments)))
    print("resume test passed")

    # cleanup
    try:
        os.remove(backup)
    except OSError:
        pass
