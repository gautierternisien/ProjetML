from section2 import *

NUM_CLASSES = 15
MODEL_PATH = "../models/vgg16_finetuned.pth"

def download_finetuned_model(model_path='../models/vgg16_finetuned.pth'):
    # Vérification si le fichier existe et s'il est valide (taille > 1Mo)
    if os.path.exists(model_path):
        if os.path.getsize(model_path) < 1024 * 1024:  # Moins de 1 Mo -> surement corrompu (HTML)
            print(f"Fichie corrompu détecté ({os.path.getsize(model_path)} bytes), suppression...")
            os.remove(model_path)
    
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print("Téléchargement du modèle fine-tuné depuis Google Drive...")
        
        # ID du fichier extrait de votre lien : 1HcDK8pRcKgn9A1j27Kgk4JORBJUQN1Fi
        file_id = '1HcDK8pRcKgn9A1j27Kgk4JORBJUQN1Fi'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        try:
            import gdown
            gdown.download(url, model_path, quiet=False)
        except ImportError:
            print("Librairie gdown non trouvée, tentative avec requests...")
            try:
                import requests
                
                def get_confirm_token(response):
                    for key, value in response.cookies.items():
                        if key.startswith('download_warning'):
                            return value
                    return None

                def save_response_content(response, destination):
                    CHUNK_SIZE = 32768
                    with open(destination, "wb") as f:
                        for chunk in response.iter_content(CHUNK_SIZE):
                            if chunk: 
                                f.write(chunk)

                URL = "https://docs.google.com/uc?export=download"
                session = requests.Session()
                response = session.get(URL, params={'id': file_id}, stream=True)
                token = get_confirm_token(response)

                if token:
                    params = {'id': file_id, 'confirm': token}
                    response = session.get(URL, params=params, stream=True)

                save_response_content(response, model_path)
            except ImportError:
                print("\nERREUR CRITIQUE : Ni 'gdown' ni 'requests' ne sont disponibles.")
                print("Veuillez installer gdown : pip install gdown")
                return

        print("Téléchargement terminé.")

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
        download_finetuned_model()

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