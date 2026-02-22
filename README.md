# AI-Generated Audio Detection (Class Project)

**Description:**
A machine learning project to detect AI-generated music by analyzing high-frequency spectral artifacts. Completed as part of the **Computer Music** course for Music Engineering at **Politecnico di Milano**.

---

## Overview
This project implements a binary classifier to distinguish between **Real** and **AI Generated** music. It leverages digital signal processing (DSP) techniques to extract a custom spectral feature ("fakeprint") that highlights generation artifacts in the frequency spectrum.

## Key Features

*   **Feature Extraction:** Implemented a `fakeprint` function that isolates high-frequency spectral peaks by subtracting the signal's lower envelope.
*   **Data Visualization:** Analyzed STFT spectrograms to identify visual differences between real and synthetic audio.
*   **Machine Learning:** Trained a simple Logistic Regression classifier with class balancing to handle dataset skew.
*   **Metadata Analysis:** Used `mutagen` to extract metadata from correctly classified real tracks.

## Tech/Libraries Used
*   **Python** (NumPy, Pandas)
*   **Audio Processing:** Librosa
*   **Machine Learning:** Scikit-learn
*   **Visualization:** Matplotlib
*   **Metadata:** Mutagen
