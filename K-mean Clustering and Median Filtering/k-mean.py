"""
K-Means Segmentation Problem
(Due date: Nov. 25, 11:59 P.M., 2019)
The goal of this task is to segment image using k-means clustering.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are allowed to add your own functions if needed.
You should design you algorithm as fast as possible. To avoid repetitve calculation, you are suggested to depict clustering based on statistic histogram [0,255]. 
You will be graded based on the total distortion, e.g., sum of distances, the less the better your clustering is.
"""


import utils
import numpy as np
import json
import time


def kmeans(img,k):
    """
    Implement kmeans clustering on the given image.
    Steps:
    (1) Random initialize the centers.
    (2) Calculate distances and update centers, stop when centers do not change.
    (3) Iterate all initializations and return the best result.
    Arg: Input image;
         Number of K. 
    Return: Clustering center values;
            Clustering labels of all pixels;
            Minimum summation of distance between each pixel and its center.  
    """
    # TODO: implement this function.
    np.seterr(over='ignore')
    labels_img = np.zeros(img.shape)  # initialize labels
    pixel_comb = []  # store possible pixel pair combinations
    cluster_result = {}  # store centers and distance
    result_point = []  # store center and distance for all pairs
    img_pixel_unique = np.unique(img)  # identify unique pixels from the image
    img_pixel_unique = np.random.choice(img_pixel_unique, 25)
    # creating possible pairs
    unorderedPairGenerator = ((x, y) for x in img_pixel_unique for y in img_pixel_unique if x < y)
    for pair in unorderedPairGenerator:
        pixel_comb.append(pair)

    for p in range(len(pixel_comb)):
        match_found = False
        new_center = pixel_comb[p]
        prev_center = pixel_comb[p]
        while (match_found != True):
            clusters0 = []
            clusters1 = []
            for i in range(len(img)):
                for j in range(len(img[0])):

                    distance_c0 = int(abs(new_center[0] - img[i][j]))
                    distance_c1 = int(abs(new_center[1] - img[i][j]))

                    if distance_c0 <= distance_c1:
                        clusters0.append(img[i][j])  # building first cluster
                        labels_img[i][j] = 0

                    elif distance_c0 > distance_c1:
                        clusters1.append(img[i][j])  # building second cluster
                        labels_img[i][j] = 1
            if len(clusters0) == 0 or len(clusters1) == 0:
                continue
            prev_center = new_center
            new_center = [int(np.mean(clusters0)), int(np.mean(clusters1))]
            if prev_center == new_center:
                sum_c0 = 0
                sum_c1 = 0
                match_found = True

                for m in range(len(clusters0)):
                    distancek1 = abs(clusters0[m] - new_center[0])
                    sum_c0 += distancek1  # distance from center to other cluster points
                for e in range(len(clusters1)):
                    distancek2 = abs(clusters1[e] - new_center[1])  # distance from center to other cluster points
                    sum_c1 += distancek2
                sum_c0_c1 = sum_c0 + sum_c1
                cluster_result["distance"] = sum_c0_c1
                cluster_result["center"] = new_center
                result_point.append(cluster_result)
    distance_only = [g['distance'] for g in result_point]

    # finding point pair with minimum distance
    center_final = []
    distance = 0
    for k in range(len(result_point)):
        if result_point[k].get("distance") == (min(distance_only)):
            center_final = result_point[k].get("center")
            distance = result_point[k].get("distance")
            break

    return result_point[k].get("center"), labels_img, int(result_point[k].get("distance"))


def visualize(centers,labels):
    """
    Convert the image to segmentation map replacing each pixel value with its center.
    Arg: Clustering center values;
         Clustering labels of all pixels. 
    Return: Segmentation map.
    """
    # TODO: implement this function.
    labels[labels == 0] = centers[0]
    labels[labels == 1] = centers[1]  # assigning centroid pixel values
    return labels.astype(np.uint8)

     
if __name__ == "__main__":
    img = utils.read_image('lenna.png')
    k = 2

    start_time = time.time()
    centers, labels, sumdistance = kmeans(img,k)
    result = visualize(centers, labels)
    end_time = time.time()

    running_time = end_time - start_time
    print(running_time)

    centers = list(centers)
    with open('results/task1.json', "w") as jsonFile:
        jsonFile.write(json.dumps({"centers":centers, "distance":sumdistance, "time":running_time}))
    utils.write_image(result, 'results/task1_result.jpg')
