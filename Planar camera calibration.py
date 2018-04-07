import cv2
import numpy as np
import random
import math
import warnings
from glob import glob


def preview(name,ig):
	cv2.imshow(name,ig)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.waitKey(1)
	pass


def parseData(points_len,basepath="./imagePt",worldpath="./worldPt.txt", ext=".txt",imgs_len=3):

    img_points = []
    for i in range(1, imgs_len+1):
        img_points.append(np.loadtxt(basepath + str(i) + ext, usecols=(0, 1)).reshape((points_len, 2)))

    return np.loadtxt(worldpath, usecols=(0, 1)).reshape((points_len, 2)),img_points


def getTestData(points_len,imagesPath ='../data/calibration/cc-imagePt*.txt',worldPath  = '../data/calibration/cc-worldPt.txt'):
	img_points = []
	imagesPt = glob(imagesPath)
	imagesPt = sorted(imagesPt, key= lambda x: (len(x),x))
	
	for path in imagesPt:
		img_points.append(np.loadtxt(path, usecols=(0, 1),skiprows=1).reshape((points_len, 2)))

	return np.loadtxt(worldPath, usecols=(0, 1),skiprows=1).reshape((points_len, 2)), img_points


def writeAll(world_points,image_points):
	for i,img in enumerate(image_points):
		open('3d-2d_'+str(i+1)+'.txt', 'w').close()

		file = open('3d-2d_'+str(i+1)+'.txt', 'w')
		# imagePt.write(str(len(img)) + '\n')

		for wp,ip in zip(world_points[0],img):
			ip = ["{0:.4f}".format(float(k)) for k in ip[0]]
			file.write(str(wp[0]) + ' ' + str(wp[1]) + ' ' + str(wp[2]) + ' ' + str(ip[0]) + ' ' + str(ip[1]) +  '\n')
		file.close()


def writeWorld(world_points):
	open('worldPt.txt', 'w').close()

	worldPt = open("worldPt.txt","w")
	# worldPt.write(str(len(world_points[0])) + '\n')

	for wp in world_points[0]:
		wp = ["{0:.2f}".format(float(w)) for w in wp]
		worldPt.write(str(wp[0]) + ' ' + str(wp[1]) + ' ' + str(wp[2]) + '\n')

	worldPt.close()


def writeImages(image_points):
	for i,img in enumerate(image_points):
		open('imagePt'+str(i+1)+'.txt', 'w').close()

		imagePt = open('imagePt'+str(i+1)+'.txt',"w")
		# imagePt.write(str(len(img)) + '\n')

		for ip in img:
			ip = ["{0:.4f}".format(float(k)) for k in ip[0]]
			imagePt.write(str(ip[0]) + ' ' + str(ip[1]) +  '\n')
		imagePt.close()


def ExtractFeature(debug):
	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((6*7,3), np.float32)
	objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	images = glob('../data/*.jpg')

	for fname in images:
	    img = cv2.imread(fname)
	    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	    # Find the chess board corners
	    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

	    # If found, add object points, image points (after refining them)
	    if ret == True:
	        objpoints.append(objp)

	        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
	        imgpoints.append(corners2)

	        # Draw and display the corners
	        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
	        cv2.imshow('img',img)
	        cv2.waitKey(600)

	cv2.destroyAllWindows()

	#get points
	writeWorld(objpoints)
	writeImages(imgpoints)
	writeAll(objpoints,imgpoints)

	#get txt to array

	wp,ips = parseData(len(objpoints[0]))


	if not debug:
		Calibrate(wp,ips)
	else:
		TestCalibration(objpoints,imgpoints,gray)
	pass


def TestCalibration(objpoints,imgpoints,gray):
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
	np.set_printoptions(formatter={'float': '{:g}'.format})
	print('\nIntrinsic K* =')
	print(np.array(mtx))


def computeMSE(world_points,img_points,H):
	MSE = 0
	M1_T = H[0,:].T
	M2_T = H[1,:].T
	M3_T = H[2,:].T
	n = len(world_points)
	for wp,ip in zip(world_points,img_points):
		mse_x = ip[0] - (M1_T.dot(wp)/M3_T.dot(wp))
		mse_y = ip[1] - (M2_T.dot(wp)/M3_T.dot(wp))
		MSE += mse_x**2 + mse_y**2
	return MSE/n


def v(i, j, H):
	# compute H matrix
    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j]
    ])


def estimateHomograpy(img_point,world_point):
	A = []
	zero = np.array([0,0,0])
	for j,ip in enumerate(img_point):
		xi = ip[0]
		yi = ip[1]
		pi = world_point[j]
		r1 = np.array(np.concatenate([pi,zero ,-xi*pi]))
		r2 = np.array(np.concatenate([zero,pi ,-yi*pi]))
		A.append(r1)
		A.append(r2)

	U, s, V = np.linalg.svd(A)
	#Compute H
	H = V[-1]
	H = np.reshape(H, (-1, 3))
	return H


def translate2DH(world_points):
	pis = []
	for p in world_points:
		pis.append([p[0],p[1],1])

	pis = np.array(pis)
	return pis


# Camera Calibration
def Calibrate(world_points,img_points,RANSAC = []):
	# Deal with world point
	pis = translate2DH(world_points)

	n = len(img_points)
	H_list = []

	# Compute A homograpy
	if len(RANSAC) > 0:
		for i in range(n):
			H = ransac(img_points[i],pis,RANSAC[0],RANSAC[1],RANSAC[2],RANSAC[3])
			H_list.append(H)
	else:
		for i in range(n):
			H = estimateHomograpy(img_points[i],pis)
			H_list.append(H)

	# Compute V
	vec = []
	for H in H_list:
		vec.append(v(0, 1, H).T)
		vec.append(v(0, 0, H).T - v(1, 1, H).T)
	
	vec = np.array(vec)
	U, s, V = np.linalg.svd(vec)
	S = V[-1]

	# Compute intrinsic K*
	c1 = (S[1]*S[3] - S[0]*S[4])
	c2 = (S[0]*S[2] - (S[1]**2))
	v0 = c1/c2

	Lambda = S[5] - ((S[3]**2) + v0*c1)/S[0]

	alpha_u = np.sqrt(np.absolute(Lambda/S[0]))
	alpha_v =  np.sqrt(np.absolute(Lambda*S[0]/c2))

	s = -S[1]*(alpha_u**2)*alpha_v/Lambda
	if -1 < s < 1:
		s = 0

	u0 = s*v0/alpha_u - S[3]*(alpha_u**2)/Lambda

	np.set_printoptions(formatter={'float': '{:g}'.format})
	Kstar = np.array([
        [alpha_u, s, u0],
        [0,     alpha_v,  v0],
        [0,     0,      1]
    ])

	print('\nIntrinsic K* =')
	print(Kstar)

	# Compute external T*, R*
	print('\nExtrinsic parameters:')
	for index,H in enumerate(H_list):
		print("\nimage "+ str(index+1)+ ":")

		K_inv = np.linalg.inv(Kstar)
		h1 = H[:,0]
		h2 = H[:,1]
		h3 = H[:,2]

		alpha_abs = 1/np.linalg.norm(np.dot(K_inv,h1))
		alpha_sign = np.sign(np.dot(K_inv,h3)[2])
		alpha = alpha_abs*alpha_sign

		r1 = alpha*K_inv.dot(h1)
		r2 = alpha*K_inv.dot(h2)
		r3 = np.cross(r1,r2)

		np.set_printoptions(formatter={'float': "{0:.2f}".format})

		Tstar = alpha*np.dot(K_inv,h3)
		print('T* = ',Tstar)		

		np.set_printoptions(formatter={'float': "{0:.6f}".format})

		Rstar = np.array(
            [r1.T, r2.T, r3.T]
         ).T
		print('R* = ',Rstar)

		print('The MSE is ', computeMSE(pis,img_points[index],H))

	pass


def medianDist(world_points,img_points,H):
	RSE_list = []
	M1_T = H[0,:].T
	M2_T = H[1,:].T
	M3_T = H[2,:].T
	for wp,ip in zip(world_points,img_points):
		e_x = ip[0] - (M1_T.dot(wp)/M3_T.dot(wp))
		e_y = ip[1] - (M2_T.dot(wp)/M3_T.dot(wp))
		RSE_list.append(np.sqrt(e_x**2 + e_y**2))
	return np.median(RSE_list)


def Inlier(world_points,img_points,H,med):
	M1_T = H[0,:].T
	M2_T = H[1,:].T
	M3_T = H[2,:].T
	inliers = []
	for i in range(len(world_points)):
		wp = world_points[i]
		ip = img_points[i]
		e_x = ip[0] - (M1_T.dot(wp)/M3_T.dot(wp))
		e_y = ip[1] - (M2_T.dot(wp)/M3_T.dot(wp))
		d = np.sqrt(e_x**2 + e_y**2)
		# print(1.5*med)
		if d < 1.5*med:
			inliers.append(i)

	return inliers


def ransac(img_points,world_points, n,d_max, P, w, random_seed=1):
	best_d = 0
	best_H = None
	count = 0
	N = n

	random.seed(random_seed)
	data_len = len(world_points)
	random_list = list(np.mgrid[0:data_len])

	K = round(math.log(1-P))/(math.log(1-(w**n)))

	while K > count and count < 1000:
		# Get correlation points 3D - 2D
		indexs = random.sample(random_list,int(N))
		
		points_3d = world_points[indexs]
		points_2d = img_points[indexs]

		# Estimate Homograpy matrix
		H = estimateHomograpy(points_2d,points_3d)
		t = medianDist(points_3d,points_2d,H)

		# find all Inliers
		inliers = Inlier(world_points,img_points,H,t)
		d = len(inliers)
		# print(d)
		if d >= d_max:
			# recompute Homograpy matrix
			points_inlier_3d = world_points[inliers]
			points_inlier_2d = img_points[inliers]
			new_H = estimateHomograpy(points_inlier_2d,points_inlier_3d)

			#check best model
			if d > best_d:
				best_d = d
				best_H = new_H
		w = d/data_len
		N = random.randint(n, round(n*(1.5+math.e**(-w))))
		K = round((math.log(1-P))/(math.log(1-(w**N))))
		count +=1

	print('took iterations:', count+1, 'best H:', best_H, 'best number of inliers:', best_d)
	return best_H


def getRANSACConfig():
	config = open('RANSAC.config', 'r')
	n = int(config.readline().split('=')[1])
	d = int(config.readline().split('=')[1])
	p = float(config.readline().split('=')[1])
	w = float(config.readline().split('=')[1])
	return n,d,p,w


def main():
	def help():	
		print('\t\t\tWelcome to corner dection !!!')
		print('Here is the function we have:')
		print(' "1" - Extract Feature point and do Calibration.')
		print(' "2" - Test the Calibration of test file')
		print(' "3" - Extract Feature point and test in cv.calibrateCamera.')
		print(' "4" - Test the Calibration of noise by Applying RANSAC')
		print(' "h" - help')
		print(' "q" - quit')

	while 1:
		option = input("\n\t\tPlease select a function('q' to exit):")
		if (option == 'h'):
			help();
		elif(option == '1'):
			ExtractFeature(False)
		elif(option == '3'):
			ExtractFeature(True)
		elif(option == '2'):
			wp,ips = getTestData(121)
			Calibrate(wp,ips)
		elif(option == '4'):

			n,d,p,w = getRANSACConfig()

			wp,ips = getTestData(121)

			og = ips.copy()
			del og[0]
			#check noise 1
			print('\n\nNOISE 1 CHECKING:')
			wp1,ips1 = getTestData(121,imagesPath ='../data/noise1/cc-noise*.txt',worldPath  = '../data/noise1/cc-worldPt.txt')
			
			og.append(ips1[0])

			Calibrate(wp,og,RANSAC = [n,d,p,w])

			og = ips.copy()
			del og[0]
			# #check noise 2
			print('\n\nNOISE 2 CHECKING:')
			wp2,ips2 = getTestData(121,imagesPath ='../data/noise2/cc-noise*.txt',worldPath  = '../data/noise2/cc-worldPt.txt')

			og.append(ips2[0])
			Calibrate(wp, og, RANSAC = [n,d,p,w])
			pass

		elif(option == 'q'):
			break
		else:
			print("Please input the code again!")
		pass
	
	pass


if __name__ == '__main__':
	print(__doc__)
	warnings.filterwarnings("ignore")
	main()
