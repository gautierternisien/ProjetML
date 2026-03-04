"""
Script d'entraînement du fine-tuning VGG16 sur 15SceneData.
Gradients propagés à tout le réseau avec LR différenciés par groupe de couches.
Les poids sont sauvegardés dans ../models/vgg16_finetuned.pth.
"""
from section2_4_FineTuningVGG16 import *

EPOCHS = 20
LR     = 1e-3
BATCH  = 32


def get_train_val_loaders(batch_size, path):
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Deux ImageFolder identiques, transforms différents
    full_train = datasets.ImageFolder(path + '/train', transform=transform_train)
    full_val   = datasets.ImageFolder(path + '/train', transform=transform_val)

    # Même split reproductible sur les deux
    n_val   = int(0.2 * len(full_train))
    n_train = len(full_train) - n_val
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(42)).tolist()
    train_indices, val_indices = indices[:n_train], indices[n_train:]

    print(f"Split : {n_train} train / {n_val} val (depuis train/) — test/ réservé à l'évaluation finale")

    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_train, train_indices), batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = torch.utils.data.DataLoader(torch.utils.data.Subset(full_val,   val_indices),   batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total   += targets.size(0)

    return running_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total   += targets.size(0)

    return running_loss / len(loader), 100. * correct / total


def plot_history(history):
    epochs_range = range(1, len(history['train_acc']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, history['train_acc'],  'b-o', label='Train')
    ax1.plot(epochs_range, history['val_acc'],    'r-o', label='Val')
    ax1.set_title('Accuracy (%)') ; ax1.set_xlabel('Epoch')
    ax1.legend() ; ax1.grid(True)

    ax2.plot(epochs_range, history['train_loss'], 'b-o', label='Train')
    ax2.plot(epochs_range, history['val_loss'],   'r-o', label='Val')
    ax2.set_title('Loss') ; ax2.set_xlabel('Epoch')
    ax2.legend() ; ax2.grid(True)

    plt.suptitle(f'Fine-tuning VGG16 — nouvelle tête linéaire (15 classes) — {len(history["train_acc"])} epochs', fontsize=13)
    plt.tight_layout()
    plt.savefig('../plot/finetuning_history.png')
    print("Courbes sauvegardées : ../plot/finetuning_history.png")
    plt.show()


if __name__ == "__main__":
    path = "../data/15SceneData"

    print("Chargement des données...")
    train_loader, val_loader = get_train_val_loaders(BATCH, path)

    print(f"Construction du modèle VGG16FineTuned ({NUM_CLASSES} classes)...")
    model = build_model()  # construit avec features + classifier_base gelés par défaut

    # Dégeler tout le réseau pour propager les gradients
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables : {trainable:,} (réseau complet)")

    criterion = nn.CrossEntropyLoss()

    # LR différenciés : les couches pré-entraînées reçoivent un LR très faible
    # pour ne pas détruire les poids ImageNet (catastrophic forgetting)
    optimizer = torch.optim.Adam([
        {'params': model.features.parameters(),        'lr': LR * 0.01},  # 1e-5 — convolutions
        {'params': model.classifier_base.parameters(), 'lr': LR * 0.1},   # 1e-4 — FC pré-entraînées
        {'params': model.fc_new.parameters(),          'lr': LR},         # 1e-3 — nouvelle couche
    ])
    # CosineAnnealing : décroissance douce du LR sur toute la durée
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nEntraînement sur {EPOCHS} epochs (réseau complet, gradients propagés)...")
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  → Nouveau meilleur modèle sauvegardé avec Val Acc: {best_val_acc:.2f}%")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")


    # Sauvegarde des poids
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nPoids sauvegardés : {MODEL_PATH}")

    plot_history(history)