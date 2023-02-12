import cv2
import numpy as np


def detect_eyes(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar cascade classifier for eye detection
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    # Detect eyes in the grayscale image
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if any eyes were detected
    if len(eyes) > 0:
        # Return the first detected eye region as an image
        x, y, w, h = eyes[0]
        return image[y:y + h, x:x + w]
    else:
        return ""


def prepare_images(image):
    common_size = (500, 500)
    image = cv2.resize(image, common_size)

    return image


def find_pupil_size(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)

    # Find the contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area, largest first
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # Get the largest contour
    cnt = contours[0]

    # Get a bounding rectangle for the contour
    x, y, w, h = cv2.boundingRect(cnt)

    # Check if the aspect ratio of the bounding rectangle is approximately 1
    aspect_ratio = w / h
    if aspect_ratio < 1.2 and aspect_ratio > 0.8:
        # Calculate the center and diameter of the pupil
        center = (x + w // 2, y + h // 2)
        diameter = min(w, h)

        # Draw the circle on the image
        cv2.circle(img, center, diameter // 2, (0, 255, 0), 2)

    # Show the image
    cv2.imshow("Detected Pupil", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the diameter of the pupil
    print("Diameter of the pupil:", diameter)


def detect_iris_shape(image):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Detect edges using the Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Find the contours in the image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(largest_contour)

    # Extract the center and axes of the ellipse
    center, axes, angle = ellipse

    # Extract the major and minor axis lengths
    major_axis, minor_axis = axes

    # Calculate the aspect ratio of the ellipse
    aspect_ratio = major_axis / minor_axis

    # Determine the shape of the iris based on the aspect ratio
    if aspect_ratio > 1.3:
        iris_shape = "Circular"
    elif aspect_ratio > 1.0:
        iris_shape = "Spatulate"
    else:
        iris_shape = "Oval"
    print("Iris shape is: " + iris_shape)


def detect_and_compare_irises_orb(iris1, iris2):
    # Resizing and converting into grayscale
    gray_iris1 = prepare_images(iris1)
    gray_iris2 = prepare_images(iris2)

    orb = cv2.ORB_create()

    # Detect keypoints and compute ORB descriptors for both images
    kp1, des1 = orb.detectAndCompute(gray_iris1, None)
    kp2, des2 = orb.detectAndCompute(gray_iris2, None)

    # Create a brute-force matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the descriptors
    matches = bf.match(des1, des2)

    # Sort the matches in order of increasing distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the matches and count the number of good matches
    img_matches = cv2.drawMatches(gray_iris1, kp1, gray_iris2, kp2, matches[:10], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    num_matches = len(matches)
    cv2.imshow("matches ORB", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("ORB number of matches: " + str(num_matches))

    if num_matches >= 100:
        print('ORB: The irises match.')
    else:
        print('ORB: The irises do not match.')


def detect_and_compare_irises_sift(iris1, iris2):
    # Resize the iris images to a common size
    gray_iris1 = prepare_images(iris1)
    gray_iris2 = prepare_images(iris2)

    # Detect key points and compute SIFT descriptors for both iris images
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_iris1, None)
    kp2, des2 = sift.detectAndCompute(gray_iris2, None)

    # Convert the descriptors to a common type
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    # Match the SIFT descriptors between the two iris images
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Filter the matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    num_matches = len(good_matches)

    # Return the number of good matches as a similarity score
    print("SIFT number of matches: " + str(num_matches))

    if num_matches >= 40:
        print('SIFT: The irises match.')
    else:
        print('SIFT: The irises do not match.')


img1 = cv2.imread('Images/Iris5.jpg')
img2 = cv2.imread('Images/iris5.jpg')

eye1 = detect_eyes(img1)

eye2 = detect_eyes(img2)

if not isinstance(eye1, str) and not isinstance(eye2, str):
    detect_and_compare_irises_orb(eye1, eye2)
    detect_and_compare_irises_sift(eye1, eye2)
    find_pupil_size(eye1)
    detect_iris_shape(eye1)
else:
    print("Eye not detected")
