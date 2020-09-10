from vidstab import VidStab
import matplotlib.pyplot as plt
import os
import cv2

stabilizer = VidStab()
stabilizer.stabilize(input_path='video_seq_1.avi', output_path='stable_video.avi')

# Image Feature Extractionn using Key Point Detection
##############################################
stabilizer = VidStab(kp_method='ORB')
stabilizer.stabilize(input_path='video_seq_1.avi', output_path='stable_video.avi', border_type='black')

stabilizer = VidStab(kp_method='FAST', threshold=42, nonmaxSuppression=False)
stabilizer.stabilize(input_path='video_seq_1.avi', output_path='stable_video.avi', border_type='black', border_size=100)
##############################################


#Plotting frame to frame transformations
stabilizer.plot_trajectory()
plt.show()

stabilizer.plot_transforms()
plt.show()


##############################################

# filled in borders
stabilizer.stabilize(input_path='video_seq_1.avi', 
                     output_path='ref_stable_video.avi', 
                     border_type='reflect')
stabilizer.stabilize(input_path='video_seq_1.avi', 
                     output_path='rep_stable_video.avi', 
                     border_type='replicate')



#################################################
#OBJECT TRACKING TEST FOR VIDEO 2 - COMMENT OUT
#################################################
# Initialize object tracker, stabilizer, and video reader
object_tracker = cv2.TrackerCSRT_create()
stabilizer = VidStab()
vidcap = cv2.VideoCapture("video_seq_1.avi")

# Initialize bounding box for drawing rectangle around tracked object
object_bounding_box = None

while True:
    grabbed_frame, frame = vidcap.read()

    # Pass frame to stabilizer even if frame is None
    stabilized_frame = stabilizer.stabilize_frame(input_frame=frame, border_size=50)

    # If stabilized_frame is None then there are no frames left to process
    if stabilized_frame is None:
        break

    # Draw rectangle around tracked object if tracking has started
    if object_bounding_box is not None:
        success, object_bounding_box = object_tracker.update(stabilized_frame)

        if success:
            (x, y, w, h) = [int(v) for v in object_bounding_box]
            cv2.rectangle(stabilized_frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

    # Display stabilized output
    cv2.imshow('Frame', stabilized_frame)

    key = cv2.waitKey(5)

    # Select ROI for tracking and begin object tracking
    # Non-zero frame indicates stabilization process is warmed up
    if stabilized_frame.sum() > 0 and object_bounding_box is None:
        object_bounding_box = cv2.selectROI("Frame",
                                            stabilized_frame,
                                            fromCenter=False,
                                            showCrosshair=True)
        object_tracker.init(stabilized_frame, object_bounding_box)
    elif key == 27:
        break

vidcap.release()
cv2.destroyAllWindows()
