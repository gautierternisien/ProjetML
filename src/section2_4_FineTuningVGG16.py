from section2 import *

NUM_CLASSES = 15
MODEL_PATH = "../models/vgg16_finetuned.pth"


class VGG16FineTuned(nn.Module):
    def __init__(self, vgg16, num_classes=NUM_CLASSES):
        super(VGG16FineTuned, self).__init__()
        self.features = nn.Sequential(*list(vgg16.features.children()))
        self.classifier_base = nn.Sequential(*list(vgg16.classifier.children())[:-1])
        self.fc_new = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_base(x)
        x = self.fc_new(x)
        return x


def get_dataset_finetuning(batch_size, path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(path + '/test', transform=transform)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader


def build_model():
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model = VGG16FineTuned(vgg16).to(DEVICE)
    return model


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    # Annoter chaque cellule
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=8)

    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vérité terrain')
    ax.set_title('Matrice de confusion — VGG16 fine-tuné (nouvelle couche linéaire 15 classes)', fontsize=14)
    plt.tight_layout()
    plt.show()


def main(path="../data/15SceneData", batch_size=32):
    if not os.path.exists(MODEL_PATH):
        print(f"Modèle introuvable : {MODEL_PATH}")
        print("Lance d'abord section2_4_train_finetuning.py pour entraîner et sauvegarder le modèle.")
        return

    print(f"Chargement du modèle depuis {MODEL_PATH}...")
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Infos sur le modèle et ses params
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres totaux      : {total_params:,}")
    print(f"Paramètres entraînés   : {trainable_params:,} ")

    print("Chargement du jeu de test...")
    test_loader = get_dataset_finetuning(batch_size, path)
    class_names = test_loader.dataset.classes

    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = (all_preds == all_targets).mean() * 100
    print(f"Accuracy : {accuracy:.2f}%")

    # Rapport par classe (précision, rappel, F1)
    print("Matrice de confusion plotée")

    # Matrice de confusion
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t][p] += 1

    plot_confusion_matrix(cm, class_names)