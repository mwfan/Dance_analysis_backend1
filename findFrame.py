import cv2
import numpy as np
import mediapipe as mp
import sys
import os
from IPython.display import Image, display
from random import randint
from sklearn.cluster import KMeans

framesToCompare = []
framesToCompare2 = []

def initializeposetools(): #initialize mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    return mp_pose, mp_drawing

def get_video_writer(image_name, video_path):
    basename = os.path.basename(video_path)  #returns the tail of the file path
    filename, extension = os.path.splitext(basename) #change extension of the file
    size = (480, 640) #set size of image

    make_directory(image_name) #making directory
    out = cv2.VideoWriter(f"{image_name}/{filename}_out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 5, size)

    print(f"{image_name}/{filename}_out.mp4")
    return out


def make_directory(name:str): #create directory
  if not os.path.isdir(name): #if directory does not exist, make directory
    os.mkdir(name)


def resize_image(image):
    h, w, _ = image.shape  #getting dimensions is height then width
    h, w = h//2, w//2
    image = cv2.resize(image, (w,h)) #changing to width then height
    return image, h, w


#improve performance
def pose_process_image(image, pose):
    #optionally mark image as not writeable to pass by reference
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #converting color
    results = pose.process(image)

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
    # Get coordinates: x and y
    a = [landmarks[p1].x, landmarks[p1].y]
    b = [landmarks[p2].x, landmarks[p2].y]
    c = [landmarks[p3].x, landmarks[p3].y]

    #calculate angle
    angle = calculate_angle(a, b, c)
    #print angles
    draw_angle(tuple(np.multiply(b, [w, h]).astype(int)), image, round(angle))
    return angle, image


def calculate_angle(a, b, c): #using 3 points + calculate the angle,
    a = np.array(a) #First
    b = np.array(b) #Mid
    c = np.array(c) #End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0: #return the smaller side of the angle
        angle = 360 - angle

    return round(angle, 1) #round to nearest ones

def draw_angle(org:tuple, image, angle):
    #font
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.4
    color = (255, 255, 255)

    thickness = 1
    image = cv2.putText(image, str(angle), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    return image

def draw_landmarks(results, mp_drawing, mp_pose, image):
    # for idx (index), x (value) in enumerate(_____):   \\storing both the index and the value
    # work w/both variables simultaneously; requires
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        if idx in [1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,29,30,31,32]:
            results.pose_landmarks.landmark[idx].visibility = 0 #remove visibility of specific landmarks

    # draw landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color = (255, 0, 0), thickness = 2, circle_radius = 2),  #customize color, etc
                              mp_drawing.DrawingSpec(color = (245, 66, 230), thickness = 2, circle_radius = 2))

    return image


def get_frames_angles(image_name:str, video_path:str) -> tuple:
    mp_pose, mp_drawing = initializeposetools()
    cap = cv2.VideoCapture(video_path)  #creating video capture object
    out = get_video_writer(image_name, video_path) #return video?
    img_count = 0
    output_images = []
    frames = []

    max_angle_right = 0
    with mp_pose.Pose(
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as pose:

        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                #if loading a video use break instead of continue
                break

            #understand what this block is doing!!!!
            if (img_count % 4 == 0):
                image, h, w = resize_image(image)
                image, results = pose_process_image(image, pose)
            # image, h, w = resize_image(image)
            # image, results = pose_process_image(image, pose)

                try:
                    landmarks = results.pose_landmarks.landmark
                    angles = plot_angles_from_frames(mp_pose, landmarks, image, h, w, max_angle_right)
                    frames.append(angles)

                    image = draw_landmarks(results, mp_drawing, mp_pose, image)
                    out.write(image)

                    cv2.imshow(image)

                    outImageFile = f"{image_name}/{image_name}{img_count}.jpg"
                    cv2.imwrite(outImageFile, image)
                    img_count += 1

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
            print(min_dist, nearest)
    return nearest

def printFrameAngles(video):
    for frame in video:
        print(frame)
    print(len(video))

def cluster(frames1, frames2): #create the two groups of clusters + compare
    student_n_cluster = int(len(frames2) / 10)
    print(student_n_cluster)
    X = np.array(frames1) #2d array from tuple list
    kmeans_1 = KMeans(n_clusters=student_n_cluster, random_state=0).fit(X)
    print(kmeans_1.labels_)

    n_cluster_coach = int(len(frames2) / 10)
    X = np.array(frames2)
    kmeans_2 = KMeans(n_clusters=n_cluster_coach, random_state = 0).fit(X)
    print(n_cluster_coach)
    print(kmeans_2.labels_)

    student_cluster = []
    start = 0
    for i in range(1, len(kmeans_1.labels_)):
        if kmeans_1.labels_[i] != kmeans_1.labels_[i - 1]:
            student_cluster.append({'label': kmeans_1.labels_[i - 1], 'start': start, 'end': i - 1})
            start = i
    else:
        student_cluster.append({'label': kmeans_1.labels_[i], 'start': start, 'end': i})

    print(student_cluster)
    used = set()
    for label in (student_cluster):
        index_student = (label['start'] + label['end']) // 2
        print('student image', index_student)

        # predict = kmeans_2.predict([frames1[index_student]])
        predict = kmeans_2.predict([frames1[index_student]])
        print('predict:', predict)
        print(frames1[index_student])
        # np.append(framesToCompare, frames1[index_student], axis=0)
        framesToCompare.append(frames1[index_student])

        indexes_frame = np.where(kmeans_2.labels_ == predict[0])
        # rand = randint(0, len(indexes_frame))
        nearest = get_nearest_neighbor(frames1[index_student], indexes_frame[0], frames2)
        print('coach', nearest)
        print(frames2[nearest])
        #np.append(framesToCompare2, frames2[nearest], axis=0)
        framesToCompare2.append(frames2[nearest])

        #display(Image(f'student/student(index_student.jpg'))
        #display(Image(f'coach/coach(nearest).jpg'))

        X = np.array([[1, 2], [1, 4], [1, 0],
                      [10, 2], [10, 4], [10, 0]])
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        # print(kmeans.labels_)
        #
        # print(kmeans.predict([[0, 0], [12, 3]]))
        #
        # print(kmeans.cluster_centers_)

def main(videoPath1, videoPath2):
    # videoPath1 = "/Users/sarahmw/PycharmProjects/project/finalproject/shorter.mov"
    # videoPath2 = "/Users/sarahmw/PycharmProjects/project/finalproject/shorter.mov"
    vid1 = get_frames_angles(image_name = "video1", video_path = videoPath1)
    vid2 = get_frames_angles(image_name = "video2", video_path = videoPath2)
    printFrameAngles(vid1)
    print("\n\n\n")
    printFrameAngles(vid2)

    cluster(vid1, vid2)

    npFramesToCompare = np.array(framesToCompare)
    npFramesToCompare2 = np.array(framesToCompare2)

    difference = abs(npFramesToCompare - npFramesToCompare2)
    # difference = (difference / npFramesToCompare2) * 100
    print(difference)

    allDifference = []

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

