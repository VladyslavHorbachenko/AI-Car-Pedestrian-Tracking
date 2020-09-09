import cv2

#Our Image
img_file = 'car.jpg'

video = cv2.VideoCapture("Pedestrians Compilation (360p).mp4")

#Training File
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

#Run forever till car stopls
while True:
        # Read the current frame
        (read_successful, frame) = video.read()

        #Safe coding

        if read_successful:
            grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break

        #Detect cars AND Pedestrians
        cars = car_tracker.detectMultiScale(grayscaled_frame)
        pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

        for (x,y,w,h) in cars:
            cv2.rectangle(frame, (x + 1, y + 2), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.imshow("Vlad", frame)

        key = cv2.waitKey(1)

        if key == 81 or key==113:
            break

video.release()
# #create openCV image
# img = cv2.imread(img_file)
#
# #conver to grayscale (needed for HAAR cascade)
# black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# #create car classifier
# car_tracker = cv2.CascadeClassifier(classifier_file)
#
# #Detect cars
# cars = car_tracker.detectMultiScale(black_n_white)
#
# #Draw rectangles around the cars:
#
# for (x,y,w,h) in cars:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
# # car1 = cars[0]
# # (x,y,w,h) = car1
# # cv2.rectangle(img, (x, y),(x + w, y + h),(0 ,0, 255), 2)
#
# #Display the image with the faces spotted
# cv2.imshow('Clever Programmer Car Detector', img)
#
# #Dont autoclose (Wait here in the code and listen for a key press)
# cv2.waitKey()
#
#
print("Code completed")