import section1
import section2
import section2_4_tune_C_SVM
import section2_4_FineTuning

def main():
    print("\n----- Section 1 -----")
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

    print("\n----- Section 2 -----")
    section2.show_activationMaps()
    X_train, y_train, X_test, y_test = section2.main()  # par défaut batch_size = 8

    print("\n----- Section 2.4 -----")
    print("Optimisation de l'hyperparamètre C du SVM via Optuna")
    section2_4_tune_C_SVM.main(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    print("\nTest du modèle VGG16 fine-tuné (poids chargés depuis ../models/vgg16_finetuned.pth)")
    section2_4_FineTuning.main()

    print("\nChangement de la couche d'extraction de relu7 à relu6")
    print("\nExploration de méthodes de réduction de dimensionnalité avant la classification")

if __name__ == "__main__":
    main()
