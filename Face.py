# 1. Setup necessary imports (ensure these are installed)
import cv2
import dlib
import numpy as np
from sklearn.cluster import KMeans
import os

# --- MODEL PATHS ---
# NOTE: Replace these with your actual local paths to the models
face_cascade_path = 'models/haarcascade_frontalface_default.xml'
predictor_path = 'models/shape_predictor_68_face_landmarks.dat'
sample_image_path = 'path/to/your/sample/face/image.jpg' # <<--- CHANGE THIS

# Initialize Dlib and OpenCV detectors
faceCascade = cv2.CascadeClassifier(face_cascade_path)
predictor = dlib.shape_predictor(predictor_path)

# -------------------------------------------------------------
## STEP 1: LOAD, RESIZE, AND DETECT FACE/LANDMARKS
# -------------------------------------------------------------

# A. Load and Preprocess Image
image = cv2.imread(sample_image_path)
image = cv2.resize(image, (500, 500)) # Standardize size
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# B. Face Detection (Haar Cascade)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(100, 100),
    flags=cv2.CASCADE_SCALE_IMAGE
)

if len(faces) == 0:
    print("No face detected.")
    exit()

(x, y, w, h) = faces[0] # Get the first detected face

# C. Landmark Detection (Dlib)
rect_dlib = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
detected_landmarks = predictor(image, rect_dlib).parts()
landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
print(f"Face box: x={x}, y={y}, w={w}, h={h}")
print("Landmarks detected.")
# 

# -------------------------------------------------------------
## STEP 2: FOREHEAD SEGMENTATION (K-MEANS)
# -------------------------------------------------------------

# A. Crop the Forehead ROI (Top 25% of the face box)
forehead_h = int(0.25 * h)
forehead = image[y:y + forehead_h, x:x + w].copy()
rows, cols, bands = forehead.shape
X = forehead.reshape(rows * cols, bands)

# B. K-Means Clustering to separate Skin (Cluster 1) from Hair/Background (Cluster 0)
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
y_kmeans = kmeans.fit_predict(X)
print("K-Means clustering complete.")

# C. Find the dominant color/cluster at the center of the forehead (likely skin)
forehead_mid = [int(cols / 2), int(rows / 2)]
# Use the cluster label of the center pixel as the 'skin' reference
center_pixel_index = forehead_mid[1] * cols + forehead_mid[0]
skin_cluster_label = y_kmeans[center_pixel_index] 

# D. Scan outward from the center to find the left and right hairline edges
lef = 0
rig = 0

# Scan left: Look for the first pixel belonging to the non-skin cluster (hair/background)
for i in range(0, cols // 2):
    current_index = forehead_mid[1] * cols + (forehead_mid[0] - i)
    if y_kmeans[current_index] != skin_cluster_label:
        lef = forehead_mid[0] - i
        break
left_edge_coord = [lef, forehead_mid[1]]

# Scan right: Look for the first pixel belonging to the non-skin cluster
for i in range(0, cols // 2):
    current_index = forehead_mid[1] * cols + (forehead_mid[0] + i)
    if y_kmeans[current_index] != skin_cluster_label:
        rig = forehead_mid[0] + i
        break
right_edge_coord = [rig, forehead_mid[1]]

# -------------------------------------------------------------
## STEP 3: MEASURE LINES AND RATIOS
# -------------------------------------------------------------

# NOTE: The coordinates need to be adjusted relative to the full image space for subtraction/measurement.
# The code logic assumes the right/left forehead points are relative to the face box's x-coordinate (x)
# Line 1: Forehead Width (Distance between right and left hairline points)
line1 = right_edge_coord[0] - left_edge_coord[0] 

# Line 2: Jaw Width (Landmark 15 minus Landmark 1)
linepointleft_2 = landmarks[1, 0] 
linepointright_2 = landmarks[15, 0] 
line2 = linepointright_2 - linepointleft_2

# Line 3: Cheek Width (Landmark 13 minus Landmark 3)
linepointleft_3 = landmarks[3, 0]
linepointright_3 = landmarks[13, 0]
line3 = linepointright_3 - linepointleft_3

# Line 4: Face Length (Chin Landmark 8's Y minus Top of Face Box Y)
linepointbottom_4 = landmarks[8, 1]
linepointtop_4 = y # The top of the Haar cascade box
line4 = linepointbottom_4 - linepointtop_4

print("\n--- MEASUREMENTS (in pixels) ---")
print(f"L1 (Forehead Width): {line1}")
print(f"L2 (Jaw Width): {line2}")
print(f"L3 (Cheek Width): {line3}")
print(f"L4 (Face Length): {line4}")


# Calculate Ratios
ratio1 = line1 / line2
ratio2 = line1 / line3
ratio3 = line2 / line3
ratio4 = line4 / line1

print("\n--- RATIOS ---")
print(f"Ratio 1 (Forehead/Jaw): {ratio1:.2f}")
print(f"Ratio 2 (Forehead/Cheek): {ratio2:.2f}")
print(f"Ratio 3 (Jaw/Cheek): {ratio3:.2f}")
print(f"Ratio 4 (Length/Forehead): {ratio4:.2f}")

# -------------------------------------------------------------
## STEP 4: CLASSIFY FACE SHAPE (HEURISTICS)
# -------------------------------------------------------------

def classify_shape(r1, r2, r3, r4):
    if r4 > 1.1:
        return "Oblong"
    elif r1 > 1.1:
        return "Oval"
    elif r1 < 0.9:
        return "Round"
    elif r2 > 1.1 and r3 > 1.1:
        return "Square"
    else:
        return "Heart"

final_prediction = classify_shape(ratio1, ratio2, ratio3, ratio4)
print(f"\nFINAL PREDICTION: {final_prediction}")