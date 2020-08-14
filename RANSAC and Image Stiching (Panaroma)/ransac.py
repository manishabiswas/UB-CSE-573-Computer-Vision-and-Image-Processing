"""
RANSAC Algorithm Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to fit a line to the given points using RANSAC algorithm, and output
the names of inlier points and outlier points for the line.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
You can use the library random
Hint: It is recommended to record the two initial points each time, such that you will Not 
start from this two points in next iteration.
"""
import random
import math


def solution(input_points, t, d, k):
    """
    :param input_points:
           t: t is the perpendicular distance threshold from a point to a line
           d: d is the number of nearby points required to assert a model fits well, you may not need this parameter
           k: k is the number of iteration times
           Note that, n for line should be 2
           (more information can be found on the page 90 of slides "Image Features and Matching")
    :return: inlier_points_name, outlier_points_name
    inlier_points_name and outlier_points_name is two list, each element of them is str type.
    For example: If 'a','b' is inlier_points and 'c' is outlier_point.
    the output should be two lists of ['a', 'b'], ['c'].
    Note that, these two lists should be non-empty.
    """
    # TODO: implement this function.
    result_point = []
    point_name = {}
    check_already_present = []
    max_count = math.factorial(len(input_points)) / (2 * (math.factorial(len(input_points) - 2)))

    def choose_sample():
        while (len(check_already_present) <= max_count):
            sample_point = random.sample(input_points, k=2)
            point_name = sample_point[0].get("name"), sample_point[1].get("name")

            if ((point_name in check_already_present) == True) or (
                    (point_name[::-1] in check_already_present) == True):  # checking uniqueness of the point pair
                continue
            check_already_present.append(point_name)
            return sample_point

    for x in range(k):
        inliner_point_list = []
        outliner_point_list = []
        result = {}
        sample_point = choose_sample()
        point_name = sample_point[0].get("name"), sample_point[1].get("name")
        point1 = sample_point[0].get("value")
        point2 = sample_point[1].get("value")
        remaining_points = [p for p in input_points if p not in sample_point]
        sum_dist = 0
        for i in range(len(remaining_points)):
            point3 = remaining_points[i].get("value")  # third point to calculate the distance
            if ((point2[0] - point1[0]) != 0):

                slope = (point2[1] - point1[1]) / (point2[0] - point1[0])  # calculating slope of the line
                intercept = point1[1] - (slope * point1[0]) # calculating intercept of the line
                distance = abs((slope * point3[0] - point3[1] + intercept) / math.sqrt((slope * slope) + 1)) # distance calculation
            else:
                distance = abs(point3[0] - point2[0])

            if distance <= t:
                inliner_point = remaining_points[i].get("name")
                inliner_point_list.append(inliner_point)
                sum_dist += distance

            else:
                outlier_point = remaining_points[i].get("name")
                outliner_point_list.append(outlier_point)

            inlier_count = len(inliner_point_list)
            outlier_count = len(outliner_point_list)

            if (inlier_count >= d):

                avg_dist = sum_dist / inlier_count
                result["inlier_list"] = (list(point_name) + inliner_point_list)
                result["Outlier_list"] = outliner_point_list
                result["Inlier Count"] = inlier_count
                result["Outlier Count"] = outlier_count
                result["Avg Distance"] = avg_dist

                result_point.append(result)
        if len(check_already_present) == max_count:
            break
    distance_only = [x['Avg Distance'] for x in result_point]

    # Searching for the point pair with minimum distance
    for k in range(len(result_point)):
        if (result_point[k].get("Avg Distance") == (min(distance_only))):
            return sorted(result_point[k].get("inlier_list")), sorted(result_point[k].get("Outlier_list"))
            break

    # raise NotImplementedError


if __name__ == "__main__":
    input_points = [{'name': 'a', 'value': (0.0, 1.0)}, {'name': 'b', 'value': (2.0, 1.0)},
                    {'name': 'c', 'value': (3.0, 1.0)}, {'name': 'd', 'value': (0.0, 3.0)},
                    {'name': 'e', 'value': (1.0, 2.0)}, {'name': 'f', 'value': (1.5, 1.5)},
                    {'name': 'g', 'value': (1.0, 1.0)}, {'name': 'h', 'value': (1.5, 2.0)}]
    t = 0.5
    d = 3
    k = 100
    inlier_points_name, outlier_points_name = solution(input_points, t, d, k)  # TODO
    assert len(inlier_points_name) + len(outlier_points_name) == 8
    f = open('./results/task1_result.txt', 'w')
    f.write('inlier points: ')
    for inliers in inlier_points_name:
        f.write(inliers + ',')
    f.write('\n')
    f.write('outlier points: ')
    for outliers in outlier_points_name:
        f.write(outliers + ',')
    f.close()
