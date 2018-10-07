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

    :param images:
    :param nx:
    :param ny:
    :return:
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

    :param img:
    :param objpoints:
    :param imgpoints:
    :return:
    '''
    return cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)


def undistort_image(img, camera_matrix, distortion_coef):
    '''

    :param img:
    :param camera_matrix:
    :param distortion_coef:
    :return:
    '''
    return cv2.undistort(img, camera_matrix, distortion_coef, \
                         None, camera_matrix)


def perspective_transform(src, dst):
    '''

    :param src:
    :param dst:
    :return:
    '''
    return cv2.getPerspectiveTransform(src, dst)


def inverse_perspective_transform(src, dst):
    '''

    :param src:
    :param dst:
    :return:
    '''
    return cv2.getPerspectiveTransform(dst, src)


def warp_image(undist_image, transform, img_size, flags=cv2.INTER_LINEAR):
    '''

    :param undist_image:
    :param transform:
    :param img_size:
    :param flags:
    :return:
    '''
    return cv2.warpPerspective(undist_image, transform, dsize=img_size, flags=flags)


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


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
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
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary


def hist(img):
    bottom_half = img[img.shape[0] // 2:, :]
    histogram = np.sum(bottom_half, axis=0)
    return histogram


def find_pixel_lane(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram.shape[:midpoint])
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


def fit_polynomial(binary_warped):
    print('hello')
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_pixel_lane(binary_warped)
    print(leftx)
    print(lefty)
    
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
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
    
    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    # Plots the left and right polynomials on the lane lines
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='red')
    plt.plot(right_fitx, ploty, color='red')
    plt.show()
    
    return out_img

def









