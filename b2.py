import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.metrics import accuracy_score

observations = list("ACGTGCGTACGTAGCGTACGTAGCGTACGT")
hidden_states_true = [0]*10 + [1]*10 + [0]*10  

nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
X = np.array([[nucleotide_map[nuc]] for nuc in observations])
y_true = np.array(hidden_states_true)

model = hmm.MultinomialHMM(n_components=2, n_iter=100, random_state=42)
model.fit(X)

y_pred = model.predict(X)
accuracy = accuracy_score(y_true, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

plt.figure(figsize=(10, 4))
plt.plot(y_true, label="True Gene Regions (0=Exon, 1=Intron)", color='blue')
plt.plot(y_pred, label="Predicted Gene Regions", color='red', linestyle='--')
plt.title("HMM Prediction vs True Gene Regions")
plt.xlabel("Nucleotide Position")
plt.ylabel("State")
plt.legend()
plt.grid(True)
plt.show()
