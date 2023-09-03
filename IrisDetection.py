import cv2
import numpy as np


def detect_eyes(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(eyes) > 0:
        x, y, w, h = eyes[0]
        cv2.imshow("Eye detected", image[y:y + h, x:x + w])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return image[y:y + h, x:x + w]
    else:
        return ""


def prepare_images(image):
    common_size = (500, 500)
    image = cv2.resize(image, common_size)

    return image


def find_pupil_size(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(gaussian, 45, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    cnt = contours[0]

    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = w / h
    if 1.2 > aspect_ratio > 0.8:
        center = (x + w // 2, y + h // 2)
        diameter = min(w, h)

        cv2.circle(img, center, diameter // 2, (0, 255, 0), 2)

        if diameter is not None:
            print("Diameter of the pupil:", diameter)

        cv2.imshow("Detected Pupil", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Pupil not found!")


def detect_iris_shape(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    ellipse = cv2.fitEllipse(largest_contour)

    center, axes, angle = ellipse

    major_axis, minor_axis = axes

    aspect_ratio = major_axis / minor_axis

    if aspect_ratio > 1.3:
        iris_shape = "Circular"
    elif aspect_ratio > 1.0:
        iris_shape = "Spatulate"
    else:
        iris_shape = "Oval"
    print("Iris shape is: " + iris_shape)

def detect_and_compare_irises_orb(iris1, iris2):
    gray_iris1 = prepare_images(iris1)
    gray_iris2 = prepare_images(iris2)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(gray_iris1, None)
    kp2, des2 = orb.detectAndCompute(gray_iris2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(gray_iris1, kp1, gray_iris2, kp2, matches[:10], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    num_matches = len(matches)

    print("ORB number of matches: " + str(num_matches))

    if num_matches >= 100:
        print('ORB: The irises match.')
    else:
        print('ORB: The irises do not match.')

    cv2.imshow("matches ORB", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_and_compare_irises_sift(iris1, iris2):
    gray_iris1 = prepare_images(iris1)
    gray_iris2 = prepare_images(iris2)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_iris1, None)
    kp2, des2 = sift.detectAndCompute(gray_iris2, None)

    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    num_matches = len(good_matches)

    print("SIFT number of matches: " + str(num_matches))

    if num_matches >= 40:
        print('SIFT: The irises match.')
    else:
        print('SIFT: The irises do not match.')


img1 = cv2.imread('Images/Iris1.jpg')
img2 = cv2.imread('Images/Iris3.jpg')

eye1 = detect_eyes(img1)

eye2 = detect_eyes(img2)

if not isinstance(eye1, str) and not isinstance(eye2, str):
    detect_and_compare_irises_orb(eye1,eye2)
else:
    print("Eye not detected")

