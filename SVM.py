from data_preprocessing import RPEDataset
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((149, 149)),
    transforms.ToTensor()
])

full_dataset = RPEDataset(dir="Good RPE Crops", transform=resize_transform)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

x_train, y_train, x_test, y_test = [], [], [], []

for data, labels in train_loader:
    data = data.view(data.size(0), -1)
    x_train.append(data.numpy())
    y_train.append(labels.numpy())

for data, labels in test_loader:
    data = data.view(data.size(0), -1)
    x_test.append(data.numpy())
    y_test.append(labels.numpy())

x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
x_test = np.concatenate(x_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
group_labels = ['Young', 'Middle', 'Old']

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=group_labels)
disp.plot(cmap='Purples', ax=ax)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
