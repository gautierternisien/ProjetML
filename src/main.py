import section1
import section2
import section2_4

def main():
    print("----- Section 1 -----")
    section1.s1("../data/cat"+".jpg")
    section1.s1("../data/cat2" + ".CR2")
    section1.s1("../data/dog" + ".jpg")
    section1.s1("../data/dog2"+".CR2")
    section1.s1("../data/dog3" + ".jpg")

    print("--- Question 4 ---")
    section1.s1_4("../data/cat"+".jpg")
    section1.s1_4("../data/cat2" + ".CR2")
    section1.s1_4("../data/dog" + ".jpg")
    section1.s1_4("../data/dog2" + ".CR2")
    section1.s1_4("../data/dog3" + ".jpg")

    print("----- Section 2 -----")
    section2.show_activationMaps()
    X_train, y_train, X_test, y_test = section2.main()  # par défaut batch_size = 8

    print("----- Section 2.4 -----")
    print("Optimisation de l'hyperparamètre C du SVM via Optuna")
    section2_4.main(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    print("Au lieu de la SVM, remplacement de la dernière couche de VGG16 une nouvelle couche fully connected de 15 classes, entraînée sur le dataset 15SceneData.")
    print("Changement de la couche d'extraction de relu7 à relu6")
    print("Exploration de méthodes de réduction de dimensionnalité avant la classification")

if __name__ == "__main__":
    main()
