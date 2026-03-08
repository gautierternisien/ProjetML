import sys
import matplotlib.pyplot as plt
import section1
import section2
import section2_4_tune_C_SVM
import section2_4_FineTuningVGG16
import section2_4_PCA_SVM
import section2_4_ResNet50

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "img":
        print("Mode avec affichage activé.")
    else:
        plt.show = lambda: None
        print("Mode sans affichage (défaut). Ajoutez 'img' en argument pour voir les graphiques.")

    print("\n-------- Section 1 : VGG16 Architecture --------")
    section1.s1("../data/cat"+".jpg")
    section1.s1("../data/cat2" + ".CR2")
    section1.s1("../data/dog" + ".jpg")
    section1.s1("../data/dog2"+".CR2")
    section1.s1("../data/dog3" + ".jpg")

    print("\n--- Question 4 ---")
    section1.s1_4("../data/cat"+".jpg")
    section1.s1_4("../data/cat2" + ".CR2")
    section1.s1_4("../data/dog" + ".jpg")
    section1.s1_4("../data/dog2" + ".CR2")
    section1.s1_4("../data/dog3" + ".jpg")

    print("\n-------- Section 2 : Transfer Learning with VGG16 on 15-Scene dataset --------")
    section2.show_activationMaps()
    X_train, y_train, X_test, y_test = section2.main()  # par défaut batch_size = 8

    print("\n-------- Section 2.4 : Going further --------")
    print("1) Optimisation de l'hyperparamètre C du SVM via Optuna")
    best_C = section2_4_tune_C_SVM.main(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    print("\n2) Exploration de méthodes de réduction de dimensionnalité avant la classification")
    section2_4_PCA_SVM.main(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, best_C=best_C)

    print("\n3) Test du modèle VGG16 fine-tuné (poids chargés depuis ../models/vgg16_finetuned.pth)")
    section2_4_FineTuningVGG16.main()

    print("\n4) Essai d'un autre modèle préentrainé pour extraire les features")
    section2_4_ResNet50.main()

if __name__ == "__main__":
    main()