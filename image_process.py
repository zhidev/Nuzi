import cv2
import numpy as np
from matplotlib import pyplot as plt


# https://stackoverflow.com/questions/28816046/
# displaying-different-images-with-actual-size-in-matplotlib-subplot
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width = im_data.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis("off")

    # Display the image.
    ax.imshow(im_data, cmap="gray")

    plt.show()


# Display except using cv2 innate members
def cv_display(window_name, in_img):
    # Display the image in a named window
    cv2.imshow(window_name, in_img)
    # Wait for a key press (0 meants wait indefinitely)
    cv2.waitKey(0)
    # Destroy all windows
    cv2.destroyAllWindows()


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image


def color_remove(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([179, 50, 255])

    # 2. Filter by size (remove small noise and huge background blocks)
    mask = cv2.inRange(hsv, lower_white, upper_white)


   # 2. Filter by size (remove small noise and huge background blocks)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    final_mask = np.zeros_like(mask)
    for i in range(1, nlabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 50 < area < 5000: # Thresholds depend on text size
            final_mask[labels == i] = 255

    # 3. Apply the cleaned mask
    result = cv2.bitwise_and(img, img, mask=final_mask)
    return result

    # Display the original and the processed image (optional)
    cv2.imshow("Mask",final_mask)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def remove_contour_with_min_area(img, min_area_threshold):
    #Grayscale conversion
    gray_image = grayscale(img)
    #cv_display("Gray Image", gray_image)

    #Apply threshold to get a binary image
    ret, img_bw = cv2.threshold(gray_image, 232, 250, cv2.THRESH_BINARY)

    #Find contours
    contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Filter contours based on area (Min threshold keeping smaller objects to get rid of hp bar)
    min_area_threshold = min_area_threshold
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #If the object is larger than the threshold, paint over it to remove
        if area > min_area_threshold:
            #Fill the contour with Black (0,0,0)
            cv2.drawContours(img_bw,[cnt], -1, (0 ,0, 0), thickness=-1)
    return img_bw

img_name = "rnbimage.png"


# Opening the image
img = cv2.imread(img_name)
# Display the image in a named window


img_cnt_removed = remove_contour_with_min_area(img, 500)

cv_display("Removed?", img_cnt_removed)

inverted_img = cv2.bitwise_not(img_cnt_removed)
cv_display("Invert image", inverted_img)
        



