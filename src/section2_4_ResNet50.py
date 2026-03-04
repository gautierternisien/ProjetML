from section2 import get_dataset, extract_features, DEVICE
from imports import *


class ResNet50Features(nn.Module):
    def __init__(self, resnet50):
        super(ResNet50Features, self).__init__()
        self.features = nn.Sequential(*list(resnet50.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


def main(path="../data/15SceneData", batch_size=8):
    print("Instanciation de ResNet50")
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    print("Instanciation de ResNet50Features")
    model = ResNet50Features(resnet50).eval().to(DEVICE)

    print("Data recovery")
    train, test = get_dataset(batch_size, path)

    print("Feature extraction")
    X_train, y_train = extract_features(train, model)
    X_test, y_test = extract_features(test, model)

    print("Training the SVM")
    svm = LinearSVC(max_iter=10000)
    svm.fit(X_train, y_train)
    accuracy = svm.score(X_test, y_test)
    print('Accuracy = %.2f%%' % (accuracy * 100))