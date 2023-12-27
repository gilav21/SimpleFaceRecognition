# pip install face_recognition
# pip install numpy
# pip install scikit-learn
# pip install pickle
# pip install opencv-python
# pip install cmake

from sklearn.svm import SVC
import face_recognition
import cv2
import pickle
import os


def train_model():

    known_encodings = []
    known_names = []

    for filename in os.listdir('photos'):
        image = face_recognition.load_image_file("photos/" + filename)
        encoding_list = face_recognition.face_encodings(image)
        if encoding_list and len(encoding_list) > 0:
            encoding = face_recognition.face_encodings(image)[0]
            known_encodings.append(encoding)
            known_names.append(filename.split('.')[0])

    # Train model
    model = SVC(kernel='linear', probability=True)
    model.fit(known_encodings, known_names)

    with open('trained_svc.pkl', 'wb') as f:
        pickle.dump(model, f)


def recognize_face_live(frame, model):
    encodings = face_recognition.face_encodings(frame)
    locations = face_recognition.face_locations(frame)

    for i in range(len(locations)):
        cv2.rectangle(frame, (locations[i][3], locations[i][0]), (locations[i][1], locations[i][2]), (0, 255, 0),
                      2)
        pred = model.predict([encodings[i]])
        cv2.rectangle(frame, (locations[i][3], locations[i][0]), (locations[i][1], locations[i][2]), (0, 255, 0),
                      1)
        cv2.putText(frame, pred[0], (locations[i][3], locations[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


def recognize_face(image_path):
    with open('trained_svc.pkl', 'rb') as f:
        model = pickle.load(f)

    test_image = face_recognition.load_image_file(image_path)

    # Get encodings
    encodings = face_recognition.face_encodings(test_image)
    locations = face_recognition.face_locations(test_image)

    for i in range(len(locations)):
        cv2.rectangle(test_image, (locations[i][3], locations[i][0]), (locations[i][1], locations[i][2]), (0, 255, 0), 2)
        pred = model.predict([encodings[i]])
        cv2.rectangle(test_image, (locations[i][3], locations[i][0]), (locations[i][1], locations[i][2]), (0, 255, 0), 1)
        cv2.putText(test_image, pred[0], (locations[i][3], locations[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    rgb_img = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    cv2.imshow(image_path, rgb_img)
    cv2.waitKey(0)


def read_live():
    cap = cv2.VideoCapture(0)
    with open('trained_svc.pkl', 'rb') as f:
        model = pickle.load(f)
    while True:
        ret, frame = cap.read()

        if ret:
            recognize_face_live(frame, model)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


if __name__ == '__main__':
    # train_model()
    # recognize_face("three.jpg")
    read_live()
