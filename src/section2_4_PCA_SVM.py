from section2 import *

def main(path="../data/15SceneData", batch_size=8, X_train=None, y_train=None, X_test=None, y_test=None, best_C=1.0):
    if X_train is None or y_train is None or X_test is None or y_test is None:
        print("Instanciation de VGG16")
        vgg16 = models.vgg16(pretrained=True)
        print("Instanciation de VGG16relu7")
        model = VGG16relu7(vgg16).eval().to(DEVICE)
        print("Data recovery")
        train, test = get_dataset(batch_size, path)
        print("Feature extraction")
        X_train, y_train = extract_features(train, model)
        X_test, y_test = extract_features(test, model)
    else:
        print("Features reçues en paramètre, extraction ignorée.")


    n_components = 1024
    print(f"\nApplication de PCA pour réduire à {n_components} dimensions...")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    variance_expliquee = pca.explained_variance_ratio_.sum() * 100
    print(f"Variance expliquée : {variance_expliquee:.1f}%")


    print("\nEntraînement du SVM avec le meilleur C trouvé par optuna après PCA...")
    svm = LinearSVC(C=best_C, max_iter=10000)
    svm.fit(X_train_pca, y_train)
    accuracy = svm.score(X_test_pca, y_test)
    print(f'Accuracy (test) = {accuracy * 100:.2f}%')