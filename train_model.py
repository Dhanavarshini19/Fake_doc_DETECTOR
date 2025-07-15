import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils.ocr_utils import extract_text_from_image
from utils.image_utils import get_image_features
from utils.nlp_utils import clean_text, get_text_features

def extract_features(image_path):
    text = extract_text_from_image(image_path)
    cleaned = clean_text(text)
    text_feats = get_text_features(cleaned)
    image_feats = get_image_features(image_path)
    return {**text_feats, **image_feats}

def load_data():
    data = []
    labels = []
    for label, folder in enumerate(["real", "fake"]):
        dir_path = os.path.join("data", folder)
        for img_file in os.listdir(dir_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(dir_path, img_file)
                feats = extract_features(path)
                data.append(feats)
                labels.append(label)
    return pd.DataFrame(data), labels

def train_and_save_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    joblib.dump(model, "model/fake_doc_model.pkl")
    print("Model trained & saved!")
    print("Train accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))

if __name__ == "__main__":
    train_and_save_model()