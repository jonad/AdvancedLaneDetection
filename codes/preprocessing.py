import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle

def grayscale(img):
    '''

    :param img:
    :return:
    '''
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def find_corners(img, nx, ny):
    '''

    :param img:
    :param nx:
    :param ny:
    :return:
    '''
    grayscale_img = grayscale(img)
    return cv2.findChessboardCorners(grayscale_img, (nx, ny), None)


def get_img_obj_points(images, nx, ny):
    '''
    get the object point and image point given (x, y) patterns.
    :param images:
    :param nx: the number of pattern on the x-axis
    :param ny: the number of pattern in the y-axis
    :return: a list of object points and image points.
    '''
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    for fname in images:
        img = cv2.imread(fname)
        ret, corners = find_corners(img, nx, ny)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    return objpoints, imgpoints


def create_pickle_file(objpoints, imgpoints):
    points = {}
    points['objpoints'], points['imgpoints'] = objpoints, imgpoints
    with open('wide_dist.pickle', 'wb') as f:
        pickle.dump(points, f, protocol=pickle.HIGHEST_PROTOCOL)


def calibrate_camera(img, objpoints, imgpoints):
    '''
    Calculate the camera matrix and distortion coefficient given an image
    :param img: the images
    :param objpoints: list of (x, y, z) coordinate of the real world image pattern.
    :param imgpoints: list of (x, y) coordinate of the image point pattern
    :return: distortion matrix, and distortion coefficient
    '''
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    return mtx, dist


def undistort_image(img, camera_matrix, distortion_coef):
    '''
    undistort an image
    :param img: distorted image
    :param camera_matrix: camera matrix
    :param distortion_coef: distortion coefficient.
    :return: undistorted image
    '''
    return cv2.undistort(img, camera_matrix, distortion_coef, \
                         None, camera_matrix)


def perspective_transform(src, dst):
    '''
    Calculate the perspective transform from four pair of
    corresponding points
    :param src: a list of four source points.
    :param dst: a list of four destination points.
    :return: perspective transform
    '''
    return cv2.getPerspectiveTransform(src, dst)


def inverse_perspective_transform(src, dst):
    '''

    :param src:
    :param dst:
    :return:
    '''
    return cv2.getPerspectiveTransform(dst, src)


def warp_image(img, transform, img_size, flags=cv2.INTER_LINEAR):
    '''
    Apply a perspective transform to an image
    :param undist_image:
    :param transform:
    :param img_size:
    :param flags:
    :return: a transformed image
    '''
    return cv2.warpPerspective(img, transform, dsize=img_size, flags=flags)


################ Thresh
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):
    '''
    calculate directional gradient
    :param img:
    :param orient:
    :param sobel_kernel:
    :param thresh:
    :return:
    '''
    gray = grayscale(img)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output


def mag_tresh(img, sobel_kernel=3, thresh_min=0, thresh_max=255):
    '''
    calculate gradient magnitude
    :param img:
    :param sobel_kernel:
    :param thresh_min:
    :param thresh_max:
    :return:
    '''
    gray = grayscale(img)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.hypot(sobelx, sobely)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh_min=0, thresh_max=np.pi / 2):
    '''
    calculate gradient direction
    :param img:
    :param sobel_kernel:
    :param thresh_min:
    :param tresh_max:
    :return:
    '''
    gray = grayscale(img)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh_min) & (absgraddir <= thresh_max)] = 1
    
    return binary_output


def combined_threshold(image, orientations, ksize, thresh):
    '''

    :param image:git
    :param orientations:
    :param ksize:
    :param thresh:
    :return:
    '''
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh_min=thresh[0], thresh_max=thresh[1])
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh_min=thresh[0], thresh_max=thresh[1])
    mag_binary = mag_tresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi / 2))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


def detect_edges(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    '''
    Detect vertical edges in an image
    :param img: image
    :param s_thresh: color threshold
    :param sx_thresh: sobel threshold
    :return:
    '''
    img = np.copy(img)
    # convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    
    # sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
   
    return color_binary


def hist(img):
    '''
    compute the distribution of x-axis value given a binary image
    :param img:
    :return: the count of number of pixel on the x-axis
    '''
    img = np.copy(img)
    # make the image binary
    img=img/255
    bottom_half = img[img.shape[0] // 2:, :]
    histogram = np.sum(bottom_half, axis=0)
    return histogram


def find_pixel_lane(binary_warped):
    '''
    Find the lane using sliding windows
    :param binary_warped:
    :return:
    '''
    # Take a histogram of the bottom half of the image
    histogram = hist(binary_warped)
    
    # create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Hyperparameters
    nwindows = 9
    margin = 100
    minpix = 50

    # set heigh of windows
    window_height = np.int(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # current positions to be updated later
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty list
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
        
        # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped, ym_per_pix=30/720,  xm_per_pix = 3.7 / 700):
   '''
   Fit the polynomial on a binary image
   :param binary_warped:
   :param ym_per_pix:
   :param xm_per_pix:
   :return:
   '''
   # Find our lane pixel
   leftx, lefty, rightx, righty, out_img = find_pixel_lane(binary_warped)
   # Fit a second order polynomial to each using `np.polyfit` in pixel
   left_fit = np.polyfit(lefty, leftx, 2)
   right_fit = np.polyfit(righty, rightx, 2)
   # Colors in the left and right lane regions
   out_img[lefty, leftx] = [255, 0, 0]
   out_img[righty, rightx] = [0, 0, 255]
   # fit a second order polynomial in meter
   left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
   right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
   
   return out_img, left_fit, right_fit, left_fit_cr, right_fit_cr

def plot_polynomial(img, out_img, left_fit, right_fit):
    '''
    Plot the fitted polynomial on the image
    :param img:
    :param out_img:
    :param left_fit:
    :param right_fit:
    :return:
    '''
    binary_warped = img[:, :, 0]
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty
    
    # Plots the left and right polynomials on the lane lines
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Binary Image', fontsize=50)
    ax2.imshow(out_img)
    ax2.plot(left_fitx, ploty, color='yellow')
    ax2.plot(right_fitx, ploty, color='yellow')
    ax2.set_title('Image with the fitted polynomial', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    
    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.show()
    
    

def hist(img):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0] // 2:, :]
    
    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram


# # Create histogram of image binary activations
# histogram = hist(img)
#
# # Visualize the resulting histogram
# plt.plot(histogram)

def calculate_curvature_offset(binary_warped, left_fit, right_fit, ym_per_pix = 30/720, xm_per_pix = 3.7 / 700):
    '''
    
    :param ploty:
    :param left_fit:
    :param right_fit:
    :return:
    '''
    
    # generate y value
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    y_eval = np.max(ploty)
    
    # curvature
    left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit[0])

    mean_curverad = np.mean([left_curverad, right_curverad])
    
    # vehicle position
    left_fitx = left_fit[0] * (y_eval * ym_per_pix) ** 2 + left_fit[1] * (y_eval * ym_per_pix) + left_fit[2]
    right_fitx = right_fit[0] *(y_eval * ym_per_pix) ** 2 + right_fit[1] * (y_eval * ym_per_pix) + right_fit[2]
    
    center_lane = np.mean([left_fitx, right_fitx])
    center_vehicle =  (binary_warped.shape[1]*xm_per_pix)/2
    offset = center_lane - center_vehicle
    
    return mean_curverad, offset

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.
    The result image is computed as follows:
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

#def detect_lane_pipeline(img):
    ###todo
    ## distortion
    ## perspective transform
    ## find pixel lane
    ## fit lane