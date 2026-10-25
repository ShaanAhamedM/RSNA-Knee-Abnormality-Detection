import json

code = """import os
import pandas as pd
import numpy as np

CLASS_PRIORS = {
    'ACL': 0.414, 'MCL': 0.155, 'Medial Meniscus': 0.448, 'Lateral Meniscus': 0.397,
    'Medial OA': 0.259, 'Lateral OA': 0.190, 'PF OA': 0.362, 'Effusion': 0.603,
    'Synovitis': 0.466, "Baker's": 0.207, 'Contusion': 0.328, 'Fracture': 0.310
}
TARGET_COLUMNS = list(CLASS_PRIORS.keys())

# 1. Search dynamically for sample_submission.csv or test.csv anywhere in /kaggle/input
sub_path = None
test_path = None

if os.path.exists('/kaggle/input'):
    for root, dirs, files in os.walk('/kaggle/input'):
        for f in files:
            if f.lower() == 'sample_submission.csv':
                sub_path = os.path.join(root, f)
            elif f.lower() == 'test.csv':
                test_path = os.path.join(root, f)

print("Found sample_submission at:", sub_path)
print("Found test.csv at:", test_path)

if sub_path is not None:
    df = pd.read_csv(sub_path)
    for col in TARGET_COLUMNS:
        df[col] = CLASS_PRIORS[col]
elif test_path is not None:
    df = pd.read_csv(test_path)
    for col in TARGET_COLUMNS:
        df[col] = CLASS_PRIORS[col]
else:
    raise FileNotFoundError(
        "Competition data not found in /kaggle/input! "
        "Please click '+ Add Input' in the right sidebar and add the competition dataset."
    )

df.to_csv('submission.csv', index=False)
print('SUCCESS! submission.csv created with shape:', df.shape)
print(df.head())
"""

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# RSNA Knee Abnormality Detection — Fast Baseline Submission\n",
                "### Guarantees at least 1 valid submission on the Kaggle leaderboard before the deadline.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [code]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r"D:\MLprojs\KaggleCompetition\RSNA\kaggle_submission_package\emergency_baseline_submission.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Updated emergency baseline notebook created successfully!")
