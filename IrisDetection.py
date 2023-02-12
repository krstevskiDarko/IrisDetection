import cv2
import decimal
import numpy as np

def prepare_images(image):
    common_size = (500, 500)
    image = cv2.resize(image, common_size)

    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def detect_and_compare_irises_orb(iris1, iris2):
    iris1 = prepare_images(iris1)
    iris2 = prepare_images(iris2)

    orb = cv2.ORB_create()

    # Detect keypoints and compute ORB descriptors for both images
    kp1, des1 = orb.detectAndCompute(iris1, None)
    kp2, des2 = orb.detectAndCompute(iris2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    img_matches = cv2.drawMatches(iris1, kp1, iris2, kp2, matches[:10], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("matches ORB", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(matches) >= 70:

        print(len(matches))
        print("The eyes matched!")
    else:
        print("The eyes did not match!")


def detect_and_compare_irises_sift(iris1, iris2):

    iris1 = prepare_images(iris1)
    iris2 = prepare_images(iris2)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(iris1, None)
    kp2, des2 = sift.detectAndCompute(iris2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Filter the matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) >= 60:
        print(len(good_matches))
        print("The eyes matched!")
    else:
        print(len(good_matches))
        print("The eyes didn't match")


def detect_eyes(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar cascade classifier for eye detection
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    # Detect eyes in the grayscale image
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if any eyes were detected
    if len(eyes) > 0:
        # Return the first detected eye region as an image
        x, y, w, h = eyes[0]
        return img[y:y + h, x:x + w]
    else:
        return "Eye not detected!"


def find_pupil_size(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 30, 100)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    ellipse = cv2.fitEllipse(largest_contour)

    # Draw the ellipse on the image
    cv2.ellipse(image, ellipse, (0, 255, 0), 2)

    # Extract the center and axes of the ellipse
    center, axes, angle = ellipse

    # Extract the major and minor axis lengths
    major_axis, minor_axis = axes

    # Calculate the pupil diameter
    pupil_diameter = min(major_axis, minor_axis)
    print("Pupil diameter is: " + str(pupil_diameter))

    cv2.imshow("Iris", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img1 = cv2.imread('Iris2.jpg')
img2 = cv2.imread('iris2.jpg')

eye1 = detect_eyes(img1)
eye2 = detect_eyes(img2)

if not isinstance(eye1, str) and not isinstance(eye2, str):
    detect_and_compare_irises_orb(eye1, eye2)
else:
    print("Eye not detected")

if not isinstance(eye1, str) and not isinstance(eye2, str):
    detect_and_compare_irises_sift(eye1, eye2)
else:
    print("Eye not detected")

find_pupil_size(eye2)
