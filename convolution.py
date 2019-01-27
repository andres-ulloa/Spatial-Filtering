
import cv2 as cv
import numpy as np
import math 


def fill_kernel(kernel_mid_x, kernel_mid_y, kernel_size, image):
    
    kernel = np.zeros((kernel_size[0], kernel_size[1]), dtype = int)
    from_x = kernel_mid_x - int(math.floor(kernel_size[1]/2))
    to_x = kernel_mid_x + int(math.floor(kernel_size[1]/2))
    from_y = kernel_mid_y -  int(math.floor(kernel_size[0]/2))
    to_y = kernel_mid_y +  int(math.floor(kernel_size[0]/2))

    for i in range(from_x, to_x + 1):
        for j in range(from_y, to_y + 1):

            kernel[i - from_x, j - from_y] = image[i,j]
            
    return kernel


def get_median(vector):
    vector = np.sort(vector)
    median = vector[int(math.floor(len(vector)/2))]
    return median


def medianFiltering(filter_size, image):
    
    filtered_image = image
    num_rows = image.shape[1]
    num_cols = image.shape[0]

    #asumes the kernel is simmetric and of odd dimensions
    edge = int(math.floor(filter_size[0]/2))

    for i in range(edge, num_rows - edge):
        for j in range(edge, num_cols - edge):
            kernel = fill_kernel(i,j, filter_size, image)
            filtered_image[i,j] = get_median(kernel.flatten())
            

    return filtered_image


def get_mean(vector):
    mean = 0
    for i in range(0, len(vector)):
        mean += vector[i]
    return int(mean/len(vector))


def meanFiltering(filter_size, image):
    
    filtered_image = image
    num_rows = image.shape[1]
    num_cols = image.shape[0]

    #assumes the kernel is simmetric and of odd dimensions
    edge = int(math.floor(filter_size[0]/2))

    for i in range(edge, num_rows - edge):
        for j in range(edge, num_cols - edge):

            kernel = fill_kernel(i,j, filter_size, image)
            filtered_image[i,j] = get_mean(kernel.flatten())

    return filtered_image


def sample_gaussian(mean, x, variance = 1):
    return math.exp((-((pow(np.linalg.norm(x - mean), 2)/ (2 * variance)))))
    

#assumes odd kernel size
def computeKernelWeights(kernel):

    middle_point_x = int(math.floor(kernel.shape[0]/2))
    middle_point_y = middle_point_x

    mean = np.array([middle_point_x, middle_point_y])

    for i in range(0 , kernel.shape[0]):
        for j in range(0 , kernel.shape[1]):
            
            x = np.array([i,j])
            kernel[i,j] = kernel[i,j] * sample_gaussian(mean,x)
    
    return kernel


def weightavgFiltering(filter_size, image):

    filtered_image = image
    num_rows = image.shape[1]
    num_cols = image.shape[0]

    #assumes the kernel is simmetric and of odd dimensions
    edge = int(math.floor(filter_size[0]/2))

    for i in range(edge, num_rows - edge):
        for j in range(edge, num_cols - edge):

            kernel = fill_kernel(i,j, filter_size, image)
            kernel = computeKernelWeights(kernel)
            filtered_image[i,j] = get_mean(kernel.flatten())
            

    return filtered_image


def fillColorKernel(kernel_mid_x, kernel_mid_y, kernel_size, image):
    
    kernel_b = np.zeros((kernel_size[0], kernel_size[1]), dtype = int)
    kernel_g = np.zeros((kernel_size[0], kernel_size[1]), dtype = int)
    kernel_r = np.zeros((kernel_size[0], kernel_size[1]), dtype = int)
    kernel = list()

    from_x = kernel_mid_x - int(math.floor(kernel_size[1]/2))
    to_x = kernel_mid_x + int(math.floor(kernel_size[1]/2))
    from_y = kernel_mid_y -  int(math.floor(kernel_size[0]/2))
    to_y = kernel_mid_y +  int(math.floor(kernel_size[0]/2))
    
    for i in range(from_x, to_x):
        for j in range(from_y, to_y):

            kernel_b[i - from_x, j - from_y] = image[i,j,0]
            kernel_g[i - from_x, j - from_y] = image[i,j,1]
            kernel_r[i - from_x, j - from_y] = image[i,j,2]
    
    kernel.append(kernel_b)
    kernel.append(kernel_g)
    kernel.append(kernel_r)

    return kernel


def meancolorFiltering(filter_size, image):

    filtered_image = image
    num_rows = image.shape[1]
    num_cols = image.shape[0]

    #assumes the kernel is simmetric and of odd dimensions

    edge = int(math.floor(filter_size[0]/2))

    for i in range(edge, num_rows - edge):
        for j in range(1, num_cols - edge):

            color_kernel = fillColorKernel(i,j, filter_size, image)
            filtered_image[i,j,0] = get_mean(color_kernel[0].flatten())
            filtered_image[i,j,1] = get_mean(color_kernel[1].flatten())
            filtered_image[i,j,2] = get_mean(color_kernel[2].flatten())

    return filtered_image



def main():

    kernel_size = (3, 3)
    grey_image = cv.imread('lena.png', 0)
    noisy_grey_img = cv.imread('pepper_noise.png', 0)
    mean_filtered_image = meanFiltering(kernel_size, grey_image)
    cv.imwrite('mean_filter.png',mean_filtered_image)
    median_filtered_image = medianFiltering(kernel_size, noisy_grey_img)
    cv.imwrite('median_filter.png', median_filtered_image)
    color_image = cv.imread('lena.png')
    color_filtered_image = meancolorFiltering(kernel_size, color_image)
    cv.imwrite('mean_color.png', color_filtered_image)
    weighted_mean_image = weightavgFiltering(kernel_size, grey_image)
    cv.imwrite('weighted_mean.png', weighted_mean_image)
    



if __name__ == '__main__':
    main()