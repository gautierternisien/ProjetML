from imports import *


def prepare_input(nameim):
    # Image originale pour l'affichage (PIL)
    img_pil = Image.open(nameim).convert('RGB')

    # Prétraitement pour le calcul
    img_resized = img_pil.resize((224, 224), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255
    img_np = img_np.transpose((2, 0, 1))

    # Normalisation ImageNet
    mu = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    sigma = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img_tensor = (torch.Tensor(img_np) - mu) / sigma
    img_tensor = img_tensor.unsqueeze(0)

    return img_pil, img_tensor


def save_and_show(plt_obj, nameim, suffix):
    output_dir = '../plot'
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(nameim).split('.')[0]
    save_path = os.path.join(output_dir, f'{suffix}_{base_name}.png')

    plt_obj.savefig(save_path)
    print(f"Graphique sauvegardé : {save_path}")
    plt_obj.show()

def s1(nameim):
    img_pil, x = prepare_input(nameim)

    # Chargement des classes
    imagenet_classes = pickle.load(open('../data/imagenet_classes.pkl', 'rb'))


    # Prédiction
    vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    vgg16.eval()

    with torch.no_grad():
        y = vgg16(x)

    pred = np.argmax(y.numpy())
    class_name = imagenet_classes[pred]
    print('Predicted class: %d - %s' % (pred, class_name))

    # Affichage et Sauvegarde
    plt.figure(figsize=(8, 6))
    plt.imshow(img_pil)
    plt.title(f"Prediction: {class_name} (Class {pred})")
    plt.axis('off')

    save_and_show(plt, nameim, "prediction")


def s1_4(nameim):
    _, img_tensor = prepare_input(nameim)

    # Chargement du modèle VGG16
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg16.eval()

    # Extraction de la première couche convolutionnelle
    first_conv_layer = vgg16.features[0]

    with torch.no_grad():
        activations = first_conv_layer(img_tensor).squeeze(0).cpu().numpy()

        plt.figure(figsize=(20, 20))
        for i in range(activations.shape[0]):
            plt.subplot(8, 8, i + 1)
            plt.imshow(activations[i], cmap='viridis')
            plt.axis('off')

        plt.suptitle(f"Activations - Couche 1 - {nameim}", fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Appel de la sauvegarde factorisée
        save_and_show(plt, nameim, "activations")