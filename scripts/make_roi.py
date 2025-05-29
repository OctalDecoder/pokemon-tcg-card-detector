import cv2

# Load the original image
IMAGE_NAME = "P-A 15.png"
ROI_SNAPSHOT = [0.0, 0.080, 1.000, 0.514] # Local image
# ROI_SNAPSHOT = [0.187, 0.265, 0.817, 0.505] # Screen
SNAPSHOT_NAME = "screen.png"

img = cv2.imread(IMAGE_NAME)
if img is None:
    raise FileNotFoundError("Could not load {IMAGE_NAME} in current directory.")

# Original dimensions
h_orig, w_orig = img.shape[:2]
x1, y1, x2, y2 = ROI_SNAPSHOT

example = img[int(y1*h_orig):int(y2*h_orig), int(x1*w_orig):int(x2*w_orig)]
cv2.imshow("EXAMPLE", example)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(SNAPSHOT_NAME, example)

# Resize to 50%
resized = img
# resized = cv2.resize(img, (w_orig // 2, h_orig // 2), interpolation=cv2.INTER_AREA)
h, w = resized.shape[:2]

# Let the user draw ROI on the resized image
roi = cv2.selectROI("Draw ROI on 50% resized image", resized, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

# roi is (x, y, width, height)
x, y, rw, rh = roi

# Compute normalized coordinates (these are valid for the original image as well)
x1 = x / w
y1 = y / h
x2 = (x + rw) / w
y2 = (y + rh) / h

print("Pixel ROI on resized image:", roi)
print(f"Normalized ROI: {x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}")
