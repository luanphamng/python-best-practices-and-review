import cv2
import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import os
import time

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=True):

    X = []
    y = []

    # Loop through each person in the training set
    print ("Start training...")
    for class_dir in os.listdir(train_dir):
        print (class_dir)
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image,1,"hog")
            

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance',n_jobs=1)
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img, knn_clf=None, model_path=None, distance_threshold=0.4, face_location=None):
   
    if len(face_location) == 0:
        return []
    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=face_location)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_location))]

    # calculate scores base on the closest distances, this score is for reference only.	
    scores = [round((1-closest_distances[0][i][0])*100,2) for i in range(len(face_location))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc,rec, score) if rec else ("unknown", loc, rec, score) for pred, loc, rec, score in zip(knn_clf.predict(faces_encodings), face_location, are_matches,scores)]
def main():
    model_path="trained_knn_model.clf"
    font = cv2.FONT_HERSHEY_DUPLEX
    with open(model_path, 'rb') as f:
            knn_clf_t = pickle.load(f)
    video_capture = cv2.VideoCapture(0)

    while True:
        start_grab_time = time.time()
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if frame is None:
            video_capture = cv2.VideoCapture(-1)
            continue
        
	#resize the frame to increase performance
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_locations = face_recognition.face_locations(rgb_small_frame,1,"hog")

	#if frame contain faces the do recognition process
        if len(face_locations) != 0:
            predictions = predict(rgb_small_frame, knn_clf=knn_clf_t, model_path="trained_knn_model.clf",distance_threshold=0.4,face_location=face_locations)
            for name, (top, right, bottom, left),rec, score in predictions:
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
    
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)
    
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 255, 255))
                
                if rec:
                    cv2.putText(frame, name+" "+str(score)+"%", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            
        end_grab_time = time.time()
        
        processing_time = end_grab_time- start_grab_time;
        
        fps = 1/processing_time
        
        cv2.putText(frame, "fps: "+str(round(fps,2)), (20,20), font, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("DemoApp",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    
def main_run():

    #uncomment this line to train the model with images inside igmages folder.
    #train(train_dir="images",n_neighbors=2,model_save_path="trained_knn_model.clf")

    main()

if __name__ == "__main__":
    main_run()

    
    
