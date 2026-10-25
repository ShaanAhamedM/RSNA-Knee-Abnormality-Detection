import json

py_path = r"D:\MLprojs\KaggleCompetition\RSNA\kaggle_submission_package\kaggle_inference_notebook.py"
ipynb_path = r"D:\MLprojs\KaggleCompetition\RSNA\kaggle_submission_package\kaggle_inference_notebook.ipynb"

with open(py_path, 'r', encoding='utf-8') as f:
    code = f.read()

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# RSNA Knee Abnormality Detection — Grandmaster Inference Notebook\n",
                "### Multi-Plane 2.5D MIL Vision Ensemble with Report-Supervised Knowledge & Prior Calibration\n",
                "\n",
                "- **Hardware:** NVIDIA GPU (P100 or T4 x2)\n",
                "- **Runtime:** Completely Offline (Internet Disabled, 9-Hour Limit)\n",
                "- **Features:**\n",
                "  - Multi-plane slice sampling (middle 20%-80% depth)\n",
                "  - Medial/lateral orientation preserved (zero blind horizontal flips)\n",
                "  - 12 independent per-target optimal sub-ensembles\n",
                "  - Log-odds temperature scaling matching empirical clinical priors $\\pi_c$\n",
                "  - Incremental disk caching every 25 studies with fail-safe error handling"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install or verify timm\n",
                "!pip install --no-index --find-links=/kaggle/input/rsna-knee-wheels timm pydicom 2>/dev/null || true\n"
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
        "language_info": {
            "name": "python"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(ipynb_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully generated {ipynb_path}!")
