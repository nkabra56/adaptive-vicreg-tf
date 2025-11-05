# Adaptive VICReg (TensorFlow/Keras)

I built a unique TensorFlow implementation of VICReg and added three features that improve small batch stability:
1) **Adaptive Variance Targeting (AVT):** I replace the fixed variance floor with a robust EMA target.
2) **Scale-Invariant Covariance (SICov):** I normalize covariance by its trace and penalize distance to a scaled identity.
3) **Cosine schedules:** I warm up alignment early and increase decorrelation later.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
