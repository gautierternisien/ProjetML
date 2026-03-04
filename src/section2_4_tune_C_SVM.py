from section2 import *
import optuna
from sklearn.model_selection import cross_val_score

def main(path="../data/15SceneData", batch_size=8,
         X_train=None, y_train=None, X_test=None, y_test=None):
    """
    Si X_train/y_train/X_test/y_test sont fournis (déjà extraits par section2.main),
    on saute l'extraction des features.
    """
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

    def objective(trial):
        C = trial.suggest_float("C", 1e-1, 1e1, log=True)
        svm = LinearSVC(C=C, max_iter=10000)
        scores = cross_val_score(svm, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
        return scores.mean()

    print("\nOptimisation du paramètre C avec Optuna (20 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    best_C = study.best_params["C"]
    print(f"\nMeilleur C trouvé : {best_C:.6f}  (accuracy CV : {study.best_value:.4f})")

    print("\nEntraînement du SVM final avec le meilleur C...")
    svm = LinearSVC(C=best_C, max_iter=10000)
    svm.fit(X_train, y_train)
    accuracy = svm.score(X_test, y_test)
    print('Accuracy (test) = %f' % accuracy)

    fig_optuna = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig_optuna.set_title("Historique d'optimisation Optuna (paramètre C du SVM)")
    plt.tight_layout()
    plt.show()
