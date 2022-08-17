import cv2
import numpy as np
import mediapipe as mp
import sys
import os
import poseprocessing
from IPython.display import Image, display
from random import randint
from sklearn.cluster import KMeans

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
res, frame = cap.read()

poseprocessing.main('/Users/sarahmw/PycharmProjects/project/finalproject/pair1_1.mov', '/Users/sarahmw/PycharmProjects/project/finalproject/pair1_2.mov')

