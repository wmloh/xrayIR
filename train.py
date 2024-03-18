import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from tqdm import tqdm, trange
from sklearn.manifold import TSNE
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

from segment.utils import loadModel, segment
from triplet import TripletImageFolder

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "output/"
EMB_SIZE = 64
NUM_EPOCHS = 10

get_output_path = lambda path: os.path.join(OUTPUT_DIR, path)

print(f"DEVICE = {DEVICE}")
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

### EDA

def mask_inputs(image):
    rl_mask, ll_mask, h_mask = segment(image / 255., segmodel, device=DEVICE)
    return (image * (rl_mask | ll_mask | h_mask).astype(int) / 255.).astype(np.float32)

segmodel = loadModel("models/weights.pt", DEVICE)
image = cv2.imread("sample.jpeg", 0)

plt.figure(figsize=(9, 5))

plt.subplot(121)
plt.imshow(image, "binary")
plt.axis("off")

plt.subplot(122)
plt.imshow(mask_inputs(image), "binary")
plt.axis("off")

plt.tight_layout()
plt.savefig(get_output_path("original_masked_sample.png"), dpi=200)
plt.close()

### Model training

# https://huggingface.co/microsoft/dit-large-finetuned-rvlcdip
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    processor = AutoImageProcessor.from_pretrained("microsoft/dit-large-finetuned-rvlcdip")
    model = AutoModelForImageClassification.from_pretrained("microsoft/dit-large-finetuned-rvlcdip")

model.classifier = nn.Linear(1024, EMB_SIZE)

triplet_ds = TripletImageFolder(root="data/train", transform=transforms.Compose([transforms.Grayscale(),
                                                                                 np.asarray,
                                                                                 mask_inputs,
                                                                                 transforms.ToPILImage(),
                                                                                 transforms.Grayscale(3),
                                                                                 transforms.Resize((112, 112)),
                                                                                 transforms.ToTensor()]))
triplet_dl = DataLoader(triplet_ds, batch_size=16, shuffle=True)

optim = torch.optim.Adam(model.parameters(), lr=8e-4)
loss_fn = nn.TripletMarginLoss()

model.train()
model.to(DEVICE)

for e in trange(NUM_EPOCHS):
    for i, (anc, pos, neg) in enumerate(triplet_dl):
        anc = anc.to(DEVICE)
        pos = pos.to(DEVICE)
        neg = neg.to(DEVICE)
        loss = loss_fn(model(anc).logits, model(pos).logits, model(neg).logits)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
    
model.eval()
model.cpu()

torch.save(model, get_output_path("embedding_model.pt"))

### Visualization

test_dataset = ImageFolder("data/test", transform=transforms.Compose([transforms.Grayscale(),
                                                                  np.asarray,
                                                                  mask_inputs,
                                                                  transforms.ToPILImage(),
                                                                  transforms.Grayscale(3),
                                                                  transforms.Resize((112, 112)),
                                                                  transforms.ToTensor()]))
test_dl = DataLoader(test_dataset, batch_size=128, shuffle=True)

X, y = next(iter(test_dl))
with torch.no_grad():
    X_emb = model(X).logits
X_red = TSNE(perplexity=25).fit_transform(X_emb.numpy())

plt.scatter(*X_red[y == 0].T, label="normal")
plt.scatter(*X_red[y == 1].T, label="pneumonia")
plt.legend()
plt.axis("off")

plt.tight_layout()
plt.savefig(get_output_path("sample_embeddings.png"), dpi=200)
plt.close()

