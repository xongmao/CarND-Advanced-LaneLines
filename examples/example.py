def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


def cal_undistort(img, objpoints, imgpoints):
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Undistort using mtx and dist
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist

def col_grad2bin(imgs, file):
    for fname in imgs:
        img = cv2.imread(fname)
        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        # Convert to grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) 
        abs_sobelx = np.absolute(sobelx) 
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        # Threshold x gradient
        thresh_min = 20
        thresh_max = 100
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        # Threshold color channel
        s_thresh_min = 170
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        
        try:
            if not os.path.exists(file):
                os.makedirs(file)
            file_suffix = os.path.splitext(img)[1]
            file_name = os.path.splitext(img)[0]
            filename = '{}{}{}{}'.format(file, os.sep, file_name, file_suffix)
            cv2.imwrite(filename, combined_binary)
        except IOError as e:
            print("IOError")
        except Exception as e:
            print("Exception")
        
def corners_unwarp(imgs, file): 
    for fname in imgs:
        img = cv2.imread(fname) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = (gray.shape[1], gray.shape[0])
        # For source points I chose four points
        src = np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
            [((img_size[0] / 6) - 10), img_size[1]],
            [(img_size[0] * 5 / 6) + 60, img_size[1]],
            [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
        # For destination points I chose four points
        dst = np.float32(
            [[(img_size[0] / 4), 0],
            [(img_size[0] / 4), img_size[1]],
            [(img_size[0] * 3 / 4), img_size[1]],
            [(img_size[0] * 3 / 4), 0]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, img_size)
        try:
            if not os.path.exists(file):
                os.makedirs(file)
            file_suffix = os.path.splitext(img)[1]
            file_name = os.path.splitext(img)[0]
            filename = '{}{}{}{}'.format(file, os.sep, file_name, file_suffix)
            cv2.imwrite(filename, warped)
        except IOError as e:
            print("IOError")
        except Exception as e:
            print("Exception")
            
def corners_unwarp2(imgs, file): 
    for fname in imgs:
        img = cv2.imread(fname) 
        img_size = (img.shape[1], img.shape[0])
        # For source points I chose four points
        src = np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
            [((img_size[0] / 6) - 10), img_size[1]],
            [(img_size[0] * 5 / 6) + 60, img_size[1]],
            [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
        # For destination points I chose four points
        dst = np.float32(
            [[(img_size[0] / 4), 0],
            [(img_size[0] / 4), img_size[1]],
            [(img_size[0] * 3 / 4), img_size[1]],
            [(img_size[0] * 3 / 4), 0]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, img_size)
        try:
            if not os.path.exists(file):
                os.makedirs(file)
            file_suffix = os.path.splitext(img)[1]
            file_name = os.path.splitext(img)[0]
            filename = '{}{}{}{}'.format(file, os.sep, file_name, file_suffix)
            cv2.imwrite(filename, warped)
        except IOError as e:
            print("IOError")
        except Exception as e:
            print("Exception")