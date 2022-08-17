import os
import glob

from moviepy.editor import *

fps = 24

# file_list = sorted(glob.glob('dance_1/*.jpg'))  # Get all the jpgs in the current directory
clips = []

for m in sorted(glob.glob('dance_1/*.jpg')):
    print(m)
    clips.append(ImageClip(m).set_duration(0.5))
# clips = [ImageClip(m).set_duration(0.05)
#          for m in file_list]

concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile("test.mp4", fps=fps)