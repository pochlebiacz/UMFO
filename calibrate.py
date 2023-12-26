import cv2 as cv
import numpy as np
import glob

# Oparte w dużej mierze na https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html, jakby ktoś chciał poczytać

# Wymiary szachownicy
CHECKERBOARD = (6, 9)

# Maksymalna liczba iteracji 30 lub epsilon <= 0,001
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Punkty układu lokalnego
local_points = np.zeros((6*9, 3), np.float32)
local_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Punkty układu globalnego i rzut na płaszczyznę obrazu
glob_points = []
projected_points = []

# Wczytanie zdjęć
images = glob.glob('images/*.jpg')

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Wykrycie szachownicy _ zwraca boolean czy wykryto i położenie wierzchołków na zdjęciu
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        glob_points.append(local_points)

        # Poprawa położenia wierzchołków z dokładnością sub-pixel
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        projected_points.append(corners2)

        # Wizualizacja działania algorytmów
        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(300)

# Kalibracja
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(glob_points, projected_points, gray.shape[::-1], None, None)

# Zapisać macierz kalibracji parametrów wewnętrznych
count = 0
np.save('intrisics_matrix', mtx)

# Usuwanie dystorcji radialnej i stycznej
for image in images:
    img = cv.imread(image)
    h, w = img.shape[:2]

    # Zwraca macierz parametrów wewnętrznych na podstawie parametru skalowania
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Usunięcie zniekształceń
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # przycięcie obrazu
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite('corrected/' + str(count) + '.jpg', dst)

    count = count + 1


# Mean re-projection error -> nie istotne, ale też nie działa
#mean_error = 0
#for i in range(len(projected_points)):
#   imgpoints2, _ = cv.projectPoints(projected_points[i], rvecs[i], tvecs[i], mtx, dist)
#   error = cv.norm(glob_points[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
#   mean_error += error
#print("total error: {}".format(mean_error / len(projected_points)))
