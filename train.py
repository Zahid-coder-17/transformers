import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gpt import BigramLanguageModel
from data.download import dataset
from tokenization.character import get_batch, decode, vocab_size


import os
import matplotlib.pyplot as plt

model = BigramLanguageModel()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
max_iter = 5000

losses = []

print("Starting Bigram Model Training...")
for epoch in range(max_iter):  
    xb, yb = get_batch("train")
    optimizer.zero_grad()
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()
    
    loss_val = loss.item()
    losses.append(loss_val)
    
    if epoch % 500 == 0 or epoch == max_iter - 1:
        print(f"Epoch {epoch:4d} | Loss: {loss_val:.4f}")

# Save loss visualization curve
os.makedirs("assets", exist_ok=True)
plt.style.use('dark_background')
plt.figure(figsize=(10, 5), dpi=300)

# Plot raw loss and moving average
epochs = list(range(max_iter))
plt.plot(epochs, losses, alpha=0.35, color='#4A90E2', label='Raw Iteration Loss')

# Calculate 50-step moving average for clean visualization
window_size = 50
moving_avg = [sum(losses[max(0, i-window_size):i+1])/len(losses[max(0, i-window_size):i+1]) for i in range(len(losses))]
plt.plot(epochs, moving_avg, color='#61DAFB', linewidth=2.5, label='Moving Average (Window=50)')

plt.title('Bigram Language Model Training Loss Curve (TinyStories)', fontsize=14, fontweight='bold', pad=15, color='white')
plt.xlabel('Training Iteration / Epoch', fontsize=12, labelpad=10)
plt.ylabel('Cross-Entropy Loss', fontsize=12, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(frameon=True, facecolor='#1E1E1E', edgecolor='none')

# Add key metric annotations
initial_loss = losses[0]
final_loss = losses[-1]
plt.annotate(f'Start: {initial_loss:.2f}', xy=(0, initial_loss), xytext=(200, initial_loss + 0.3),
             arrowprops=dict(facecolor='#FF6B6B', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, fontweight='bold', color='#FF6B6B')

plt.annotate(f'Final: {final_loss:.2f}', xy=(max_iter-1, final_loss), xytext=(max_iter-1200, final_loss + 0.8),
             arrowprops=dict(facecolor='#4EBD40', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, fontweight='bold', color='#4EBD40')

plt.tight_layout()
plot_path = "assets/loss_curve.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved training plot to {plot_path}")

print("\n--- Model Text Generation Sample ---")
idx = torch.zeros((1, 1), dtype=torch.long)
generated_text = model.generate(idx, 150)
print(decode(generated_text[0].tolist()))

        
        