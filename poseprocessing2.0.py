import cv2
import mediapipe as mp
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
import json

framesToCompare = []
framesToCompare2 = []


def set_up_pose_detection_model():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    return mp_pose, mp_drawing


def get_video_writer(image_name, video_path):
    basename = os.path.basename(video_path)
    filename, extension = os.path.splitext(basename)
    size = (480, 640)
    make_directory(image_name)
    out = cv2.VideoWriter(f"{image_name}/{filename}_out.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, size)
    print(f"{image_name}/{filename}_out.avi")
    return out


def make_directory(name: str):
    if not os.path.isdir(name):
        os.mkdir(name)


def resize_image(image):
    h, w, _ = image.shape
    h, w = h // 2, w // 2
    image = cv2.resize(image, (w, h))
    return image, h, w


def pose_process_image(image, pose):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results


def plot_angles_from_frames(mp_pose, landmarks, image, h, w, max_angle_right=0, ):
    angles = []
    val = 50
    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                              mp_pose.PoseLandmark.LEFT_ELBOW.value,
                              mp_pose.PoseLandmark.LEFT_WRIST.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                              mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                              mp_pose.PoseLandmark.RIGHT_WRIST.value, landmarks, image, h, w - val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_HIP.value,
                              mp_pose.PoseLandmark.LEFT_KNEE.value,
                              mp_pose.PoseLandmark.LEFT_ANKLE.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_HIP.value,
                              mp_pose.PoseLandmark.RIGHT_KNEE.value,
                              mp_pose.PoseLandmark.RIGHT_ANKLE.value, landmarks, image, h, w - val)
    angles.append(angle)

    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                              mp_pose.PoseLandmark.LEFT_HIP.value,
                              mp_pose.PoseLandmark.LEFT_KNEE.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                              mp_pose.PoseLandmark.RIGHT_HIP.value,
                              mp_pose.PoseLandmark.RIGHT_KNEE.value, landmarks, image, h, w - val)
    angles.append(angle)

    angle_wrist_shoulder_hip_left, image = plot_angle(mp_pose.PoseLandmark.LEFT_WRIST.value,
                                                      mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                                                      mp_pose.PoseLandmark.LEFT_HIP.value, landmarks, image, h, w + val)

    angle_wrist_shoulder_hip_right, image = plot_angle(mp_pose.PoseLandmark.RIGHT_WRIST.value,
                                                       mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                                       mp_pose.PoseLandmark.RIGHT_HIP.value, landmarks, image, h,
                                                       w - val)

    angles.append(angle_wrist_shoulder_hip_left)
    angles.append(angle_wrist_shoulder_hip_right)

    return angles


def plot_angle(p1, p2, p3, landmarks, image, h, w):
    # Get coordinates
    a = [landmarks[p1].x,
         landmarks[p1].y]
    b = [landmarks[p2].x, landmarks[p2].y]
    c = [landmarks[p3].x, landmarks[p3].y]

    # Calculate angle
    angle = calculate_angle(a, b, c)
    # print(angle)
    draw_angle(tuple(np.multiply(b, [w, h]).astype(int)), image, round(angle))
    return angle, image


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return round(angle, 1)


def draw_angle(org: tuple, image, angle):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 0.4
    # Blue color in BGR
    color = (255, 255, 255)

    # Line thickness of 2 px
    thickness = 1

    # Using cv2.putText() method
    image = cv2.putText(image, str(angle), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return image


def draw_landmarks(results, mp_drawing, mp_pose, image):
    # do not display hand, feet
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        if idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32]:
            results.pose_landmarks.landmark[idx].visibility = 0

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    return image


def get_frames_angles(image_name: str, video_path: str) -> tuple:
    mp_pose, mp_drawing = set_up_pose_detection_model()
    cap = cv2.VideoCapture(video_path)
    out = get_video_writer(image_name, video_path)
    img_count = 0
    output_images = []
    frames = []

    max_angle_right = 0
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break
            if (img_count % 4 == 0):
                image, h, w = resize_image(image)
                image, results = pose_process_image(image, pose)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    angles = plot_angles_from_frames(mp_pose, landmarks, image, h, w, max_angle_right)
                    frames.append(angles)

                    image = draw_landmarks(results, mp_drawing, mp_pose, image)
                    out.write(image)

                    cv2.imshow(image)  # in python IDE, change cv2_imshow to cv2.imshow('title of frame/image', image)

                    outImageFile = f"{image_name}/{image_name}{img_count}.jpg"
                    cv2.imwrite(outImageFile, image)
                except:
                    pass
            img_count += 1

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    out.release()

    return frames


def get_nearest_neighbor(image, indexes, frames):
    a = np.array(image)
    min_dist = sys.maxsize
    nearest = indexes[0]
    for idx in indexes:
        b = np.array(frames[idx])
        dist = np.linalg.norm(a - b)
        if min_dist > dist:
            nearest = idx
            min_dist = dist
            # print(min_dist, nearest)
    return nearest


def display_frames(frames1, frames2):
    for frame in frames2:
        print(frame)
    print(len(frames2))
    print('\n\n\n')

    for frame in frames1:
        print(frame)
    print(len(frames1))


from IPython.display import Image, display
from random import randint
from sklearn.cluster import KMeans
import numpy as np


def clusters(frames1, frames2):
    # student_n_cluster = kmean_hyper_param_tuning(frames1)
    student_n_cluster = int(len(frames2) / 10)
    print(student_n_cluster)
    X = np.array(frames1)
    kmeans_student = KMeans(n_clusters=student_n_cluster, random_state=0).fit(X)
    # print(kmeans_student.labels_)

    # print(kmeans.cluster_centers_)

    # n_cluster_coach = kmean_hyper_param_tuning(frames2)
    n_cluster_coach = int(len(frames2) / 10)
    X = np.array(frames2)
    kmeans_coach = KMeans(n_clusters=n_cluster_coach, random_state=0).fit(X)
    print(n_cluster_coach)
    # print(kmeans_coach.labels_)

    student_cluster = []
    start = 0
    for i in range(1, len(kmeans_student.labels_)):
        if kmeans_student.labels_[i] != kmeans_student.labels_[i - 1]:
            student_cluster.append({'label': kmeans_student.labels_[i - 1], 'start': start, 'end': i - 1})
            start = i
    else:
        student_cluster.append({'label': kmeans_student.labels_[i], 'start': start, 'end': i})

    # for index_student in range (0,len(frames1),10):

    # print(student_cluster)
    used = set()
    for label in (student_cluster):
        index_student = (label['start'] + label['end']) // 2
        # print('student image ', index_student)
        # predict = kmeans_coach.predict([frames1[index_student]])
        predict = kmeans_coach.predict([frames1[index_student]])
        # print('predict:', predict)
        # print(frames1[index_student])
        # np.append(framesToCompare, frames1[index_student], axis = 0)
        framesToCompare.append(frames1[index_student])

        indexes_frame = np.where(kmeans_coach.labels_ == predict[0])
        # rand = randint(0,len(indexes_frame))
        nearest = get_nearest_neighbor(frames1[index_student], indexes_frame[0], frames2)
        # print('coach', nearest)
        # print(frames2[nearest])
        # np.append(framesToCompare2, frames2[nearest], axis = 0)
        framesToCompare2.append(frames2[nearest])

        # display(Image(f'student/student{index_student}.jpg'))
        # display(Image(f'coach/coach{nearest}.jpg'))

        X = np.array([[1, 2], [1, 4], [1, 0],
                      [10, 2], [10, 4], [10, 0]])
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        # print(kmeans.labels_)
        #
        # print(kmeans.predict([[0, 0], [12, 3]]))
        #
        # print(kmeans.cluster_centers_)


def main(v1_name, v2_name):
    frames1 = get_frames_angles(image_name='student',
                                video_path=v1_name)
    frames2 = get_frames_angles(image_name='coach', video_path=v2_name)

    display_frames(frames1, frames2)
    clusters(frames1, frames2)
    # print(framesToCompare)
    # print('\n\n\n')
    # print(framesToCompare2)
    #
    # print('\n\n\n')

    npFramesToCompare = np.array(framesToCompare)
    npFramesToCompare2 = np.array(framesToCompare2)

    difference = abs(npFramesToCompare2 - npFramesToCompare)
    # difference = (difference / npFramesToCompare2) * 100
    print(difference)

    allDifference = []
    # for i in difference:
    #     temp = 0
    #     for j in range(8):
    #         temp += i[j]
    #     if temp >= 20:
    #         allDifference.append(temp)
    # print(allDifference)
    # allDifference = np.array(allDifference)
    # totalFrameError  = np.sum(allDifference)
    # print(totalFrameError)
    # averageFrameError = totalFrameError / len(difference)
    # print(averageFrameError)
    errorTotal = 0
    for arr in difference:
        for dif in range(8):
            if arr[dif] >= 15:
                allDifference.append(arr[dif])

    for i in allDifference:
        errorTotal += i
    print(errorTotal)
    averageError = errorTotal / (len(difference) * 8 * 15) * 100

    print("your score: " + str(averageError))

main('/Users/sarahmw/PycharmProjects/project/finalproject/pair1_1.mov', '/Users/sarahmw/PycharmProjects/project/finalproject/pair1_2.mov')


