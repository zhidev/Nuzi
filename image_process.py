import re

import cv2
import numpy as np
import pytesseract
from PIL import Image
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
        if 50 < area < 5000:  # Thresholds depend on text size
            final_mask[labels == i] = 255

    # 3. Apply the cleaned mask
    result = cv2.bitwise_and(img, img, mask=final_mask)
    return result


def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


def remove_contour_with_min_area(img, min_area_threshold):
    # Grayscale conversion
    gray_image = grayscale(img)
    # cv_display("Gray Image", gray_image)

    # Apply threshold to get a binary image
    ret, img_bw = cv2.threshold(gray_image, 232, 250, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(
        img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter contours based on area (Min threshold keeping smaller objects to get rid of hp bar)
    min_area_threshold = min_area_threshold
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # If the object is larger than the threshold, paint over it to remove
        if area > min_area_threshold:
            # Fill the contour with Black (0,0,0)
            cv2.drawContours(img_bw, [cnt], -1, (0, 0, 0), thickness=-1)
    return img_bw

    # RegEx For scraping Levels. Fir


def is_ocr_level(input_string):
    #:param input_string: String to regex on
    # :return: true if matches regex for Levels, false if it doesn't
    # :rtype: bool

    # ^lu[a-zA-Z0-9]{1,3}$
    # is the regex for luXXX. lu is used because ocr reads the v as a u
    level_pattern = "^Lu[a-zA-Z0-9]{1,3}$"
    if re.match(level_pattern, input_string):
        return True
    else:
        return False


def is_hp_value(input_string):
    hp_pattern = "^[0-9]{1,3}/[0-9]{2,3}$"
    if re.match(hp_pattern, input_string):
        return True
    else:
        return False


def is_possible_name(input_string):
    # Pattern is just not empty string
    name_pattern = ".+"
    if re.match(name_pattern, input_string):
        return True
    else:
        return False

def resize_image(img, x,y):
    new_dimensions = (x, y) 

    h, w = img.shape[:2]
    new_width = 500
    
    # Calculate ratio and new height
    ratio = new_width / float(w)
    new_height = int(h * ratio)
    
    # Resize using calculated dimensions
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA) # Resize the image
    print("CHECKING IMAGE TYPE IN RESIZE IMG")
    check_image_type(img)
    resized_image = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_LINEAR)
    return resized_image

    # Display the result (optional)
    cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("???")
    return resized_image


def check_image_type(image):
    if isinstance(image, np.ndarray):
        print("The image is an OpenCV (NumPy array) image.")
    elif isinstance(image, Image.Image):
        print("The image is a PIL (Pillow) image.")
    else:
        print(f"The image is an unknown type: {type(image)}")

def convert_PIL_to_cv_img(pil_img):
    print("CONVERTING PIL TO CV IMG")
    if pil_img is None:
        print("Warning: Received None instead of PIL Image")
        return None
    try:
        # Perform conversion
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return cv_img 
    except Exception as e:
        print(f"Conversion failed: {e}")
        return None



def preprocess_cv_image_into_dilate_img(img):

    #ocr_result = pytesseract.image_to_string(thick_img)
    # print(ocr_result)

    blur = cv2.GaussianBlur(img, (7, 7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    dilate = cv2.dilate(thresh, kernal, iterations=6)
    return dilate

def brighten_image(img):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=50)

def dim_image(img):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=-50)


# flag = 0 means return a list of names
def ocr_image(img, flag):
    # cv_display("Original", img)
    check_image_type(img)
    #1400, 700 was the dimensions of the test image
    # img = resize_image(img, 1041, 701)
    check_image_type(img)
    img_cnt_removed = remove_contour_with_min_area(img, 2000)
    # cv_display("Pain", img_cnt_removed)
    #img_name = "rnbcode.png"
    inverted_img = cv2.bitwise_not(img_cnt_removed)
    # cv_display("Inverted", inverted_img)

    thick_img = thick_font(inverted_img)
    # cv_display("thick", thick_img)

    #ocr_result = pytesseract.image_to_string(thick_img)
    # print(ocr_result)

    imgcopy = thick_img.copy()


    # Opening the image
    #img = cv2.imread(img_name)
    # Display the image in a named window
    dilate = preprocess_cv_image_into_dilate_img(img=thick_img)

    #draws rectangles around contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
    results = []
    all_lvls = []
    all_hp_values = []
    all_possible_names = []

    hbound = 200
    wbound = 100
    found_first_entry = False
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h > hbound and w > wbound:
            roi = imgcopy[y : y + h, x : x + w]
            cv2.rectangle(imgcopy, (x, y), (x + w, y + h), (36, 255, 12), 2)
            ocr_result = pytesseract.image_to_string(roi)
            ocr_result = ocr_result.split("\n")
            for item in ocr_result:
                # if the item is a lu we append to levels string
                if is_ocr_level(item):
                    all_lvls.append(item)
                    if not found_first_entry:
                        found_first_entry = True
                        hbound = hbound + 300
                elif is_hp_value(item):
                    all_hp_values.append(item)
                else:
                    results.append(item)

    # cv_display("Image Copy:", imgcopy)
    # cv_display("ROI", roi)


    # print("All levels below:")
    # print(all_lvls)

    # print("All HP values below:")
    # print(all_hp_values)

    for item in results:
        if is_possible_name(item):
            all_possible_names.append(item)

    # print("All Name values below:")
    # print(all_possible_names)

    return_dict = {
        'ocr_name' : all_possible_names,
        'hp_value' : all_hp_values
    }

    if flag == 0:
        return all_possible_names


if __name__ == '__main__':
    
    img_name = "rnbcode.png"

    
    # # Opening the image
    img = cv2.imread(img_name)
    # # Display the image in a named window
    ocr_image(img=img)
    pass