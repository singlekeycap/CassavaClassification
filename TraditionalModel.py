# AI Use Disclosure: This code was written with the assistance of Github Copilot.  
# AI was also used for the idea of extracting features to improve the model performance.
import cv2
import numpy as np
import pandas as pd
import sklearn

def extract_features(images):
    # Extract features from the images using color spaces
    hsv = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(images, cv2.COLOR_BGR2LAB)
    features = []

    hsv_ranges = [(0, 180), (0, 256), (0, 256)]
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [256], hsv_ranges[channel])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    for channel in range(3):
        hist = cv2.calcHist([lab], [channel], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    h_step, w_step = hsv.shape[0] // 4, hsv.shape[1] // 4
    for i in range(4):
        for j in range(4):
            patch = hsv[i * h_step:(i + 1) * h_step, j * w_step:(j + 1) * w_step]
            features.extend(patch.mean(axis=(0, 1)))
            features.extend(patch.std(axis=(0, 1)))

    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    features.append(np.mean(edges > 0))
    features.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    return np.array(features, dtype=np.float32)

    
Data_Path = "C:/Users/rdsip/Downloads/cassava-leaf-disease-classification"
Image_Path = Data_Path + "/train_images"
CSV_Path = Data_Path + "/train.csv"

df = pd.read_csv(CSV_Path)
X = np.empty((len(df), 128 * 128 * 3), dtype=np.uint8)
y = np.empty(len(df), dtype=np.uint8)
count = 0
for _, row in df.iterrows():
    image_path = Image_Path + "/" + row["image_id"]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        continue

    # Need to resize for consistent input size for the model & RAM restrictions
    image_resized = cv2.resize(image, (128, 128))
    flattened_image = image_resized.flatten()
    X[count] = flattened_image
    y[count] = row["label"]
    count += 1

X = X[:count]
y = y[:count]

print(f"Total images processed: {len(X)}")
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, test_size=0.2, stratify=y, random_state=19)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, 
                                             max_depth=20,
                                             min_samples_leaf=5,
                                             max_features='sqrt',
                                             class_weight='balanced_subsample',
                                             n_jobs=-1,
                                             random_state=19)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")
print(f"Classification Report:\n{sklearn.metrics.classification_report(y_val, y_pred)}")

rng = np.random.default_rng(19)
keep_indices = []
for label in np.unique(y_train):
    label_indices = np.where(y_train == label)[0]
    if len(label_indices) > 1500:
        selected_indices = rng.choice(label_indices, size=1500, replace=False)
    else:
        selected_indices = label_indices
    keep_indices.append(selected_indices)

keep_indices = np.concatenate(keep_indices)
X_train_balanced = X_train[keep_indices]
y_train_balanced = y_train[keep_indices]

rf_balanced = sklearn.ensemble.RandomForestClassifier(n_estimators=100,
                                                     max_depth=20,
                                                     min_samples_leaf=5,
                                                     max_features='sqrt',
                                                     class_weight='balanced_subsample',
                                                     n_jobs=-1,
                                                     random_state=19)

rf_balanced.fit(X_train_balanced, y_train_balanced)

y_pred_balanced = rf_balanced.predict(X_val)
accuracy_balanced = sklearn.metrics.accuracy_score(y_val, y_pred_balanced)
print(f"Validation Accuracy (Balanced): {accuracy_balanced}")
print(f"Classification Report (Balanced):\n{sklearn.metrics.classification_report(y_val, y_pred_balanced)}")
'''
hgb = sklearn.ensemble.HistGradientBoostingClassifier(  max_iter=200,             
                                                        learning_rate=0.05,
                                                        max_leaf_nodes=31,
                                                        min_samples_leaf=20,
                                                        max_features=0.05, 
                                                        class_weight="balanced",
                                                        early_stopping=True,
                                                        random_state=19,
                                                    )

hgb.fit(X_train_balanced, y_train_balanced)

y_pred_hgb = hgb.predict(X_val)
accuracy_hgb = sklearn.metrics.accuracy_score(y_val, y_pred_hgb)
print(f"Validation Accuracy (HGB): {accuracy_hgb}")
print(f"Classification Report (HGB):\n{sklearn.metrics.classification_report(y_val, y_pred_hgb)}")
'''

# Try again with extracted features
X = np.empty((len(df), 962), dtype=np.float32)
y = np.empty(len(df), dtype=np.uint8)
count = 0
for _, row in df.iterrows():
    image_path = Image_Path + "/" + row["image_id"]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        continue

    # Need to resize for consistent input size for the model & RAM restrictions
    image_resized = cv2.resize(image, (128, 128))
    features = extract_features(image_resized)
    X[count] = features
    y[count] = row["label"]
    count += 1

X = X[:count]
y = y[:count]

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, test_size=0.2, stratify=y, random_state=19)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100,
                                             max_depth=20,
                                             min_samples_leaf=5,
                                             max_features='sqrt',
                                             class_weight='balanced_subsample',
                                             n_jobs=-1,
                                             random_state=19)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")
print(f"Classification Report:\n{sklearn.metrics.classification_report(y_val, y_pred)}")

hgb_full = sklearn.ensemble.HistGradientBoostingClassifier(max_iter=200,
                                                            learning_rate=0.05,
                                                            max_leaf_nodes=31,
                                                            min_samples_leaf=20,
                                                            max_features=0.5,
                                                            class_weight="balanced",
                                                            early_stopping=True,
                                                            random_state=19)

hgb_full.fit(X_train, y_train)

y_pred_hgb_full = hgb_full.predict(X_val)
accuracy_hgb_full = sklearn.metrics.accuracy_score(y_val, y_pred_hgb_full)
print(f"Validation Accuracy (HGB Full): {accuracy_hgb_full}")
print(f"Classification Report (HGB Full):\n{sklearn.metrics.classification_report(y_val, y_pred_hgb_full)}")

rng = np.random.default_rng(19)
keep_indices = []
for label in np.unique(y_train):
    label_indices = np.where(y_train == label)[0]
    if len(label_indices) > 1500:
        selected_indices = rng.choice(label_indices, size=1500, replace=False)
    else:
        selected_indices = label_indices
    keep_indices.append(selected_indices)

keep_indices = np.concatenate(keep_indices)
X_train = X_train[keep_indices]
y_train = y_train[keep_indices]

hgb_balanced = sklearn.ensemble.HistGradientBoostingClassifier(max_iter=200,
                                                                learning_rate=0.05,
                                                                max_leaf_nodes=31,
                                                                min_samples_leaf=20,
                                                                max_features=0.5,
                                                                class_weight="balanced",
                                                                early_stopping=True,
                                                                random_state=19)

hgb_balanced.fit(X_train, y_train)

y_pred_hgb_balanced = hgb_balanced.predict(X_val)
accuracy_hgb_balanced = sklearn.metrics.accuracy_score(y_val, y_pred_hgb_balanced)
print(f"Validation Accuracy (HGB Balanced): {accuracy_hgb_balanced}")
print(f"Classification Report (HGB Balanced):\n{sklearn.metrics.classification_report(y_val, y_pred_hgb_balanced)}")
