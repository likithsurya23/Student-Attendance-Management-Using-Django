import os
import cv2
import joblib
import numpy as np
import pandas as pd
from datetime import date, datetime
from django.contrib.auth.forms import AuthenticationForm  # Add this import
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import Http404
from django.http import HttpResponse
from sklearn.neighbors import KNeighborsClassifier
import shutil

# Set the number of images for training
nimgs = 10

# Load the face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set date format
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Ensure necessary directories exist
if not os.path.isdir('media/Attendance'):
    os.makedirs('media/Attendance')
if not os.path.isdir('media/static/faces'):
    os.makedirs('media/static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('media/Attendance'):
    with open(f'media/Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('media/static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except Exception as e:
        print(f"Error extracting faces: {e}")
        return []

def identify_face(facearray):
    model = joblib.load('media/static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('media/static/faces')
    for user in userlist:
        for imgname in os.listdir(f'media/static/faces/{user}'):
            img = cv2.imread(f'media/static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'media/static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'media/Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'media/Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'media/Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

def getallusers():
    userlist = os.listdir('media/static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for user in userlist:
        name, roll = user.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

def home(request):
    names, rolls, times, l = extract_attendance()
    indexed_attendance = [{'index': i + 1, 'name': names[i], 'roll': rolls[i], 'time': times[i]} for i in range(l)]
    return render(request, 'attendance/home.html', {'indexed_attendance': indexed_attendance, 'totalreg': totalreg(), 'datetoday2': datetoday2})

def start(request):
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('media/static'):
        return render(request, 'attendance/home.html', {'names': names, 'rolls': rolls, 'times': times, 'l': l, 'totalreg': totalreg(), 'datetoday2': datetoday2, 'mess': 'There is no trained model in the static folder. Please add a new face to continue.'})
    imgBackground = cv2.imread("background.png")
    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if frame is None:
            print("Error: No frame captured from the camera.")
            continue
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (0, 0, 255), -1)
            cv2.putText(frame, f'{identified_person}', (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow('Attendance', imgBackground)
        
        key = cv2.waitKey(1)
        if key == 27 or key == ord('O') or key == ord('o'):  # 27 is ESC key, ord('O') for 'O' key
            break

    cap.release()
    cv2.destroyAllWindows()

    names, rolls, times, l = extract_attendance()
    return render(request, 'attendance/home.html', {'names': names, 'rolls': rolls, 'times': times, 'l': l, 'totalreg': totalreg(), 'datetoday2': datetoday2})

def add(request):
    if request.method == 'POST':
        newusername = request.POST['newusername']
        newuserid = request.POST['newuserid']
        userimagefolder = 'media/static/faces/'+newusername+'_'+str(newuserid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        i, j = 0, 0
        cap = cv2.VideoCapture(0)
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = newusername+'_'+str(i)+'.jpg'
                    cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            if j == nimgs*5:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        train_model()
        names, rolls, times, l = extract_attendance()
        return render(request, 'attendance/home.html', {'names': names, 'rolls': rolls, 'times': times, 'l': l, 'totalreg': totalreg(), 'datetoday2': datetoday2})
    return redirect('home')

def add_user(request):
    if request.method == 'POST':
        newusername = request.POST['newusername']
        newuserid = request.POST['newuserid']
        userimagefolder = 'media/static/faces/' + newusername + '_' + str(newuserid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        i, j = 0, 0
        cap = cv2.VideoCapture(0)
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = newusername + '_' + str(i) + '.jpg'
                    cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            if j == nimgs * 5:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        train_model()
        return redirect('home')
    return render(request, 'attendance/add_user.html', {'totalreg': totalreg()})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                if user.is_superuser:  # Check if the user is a superuser
                    login(request, user)
                    return redirect('dashboard')  # Redirect to the dashboard after login
                else:
                    messages.error(request, "You must be a superuser to access this page.")
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Failed to ")
    else:
        form = AuthenticationForm()
    
    return render(request, 'attendance/login.html', {'form': form})

def dashboard(request):
    userlist, names, rolls, l = getallusers()
    students = [{'name': names[i], 'id': rolls[i]} for i in range(l)]
    return render(request, 'attendance/dashboard.html', {'students': students})

def get_student_by_id(student_id):
    userlist = os.listdir('media/static/faces')
    for user in userlist:
        name, id = user.split('_')
        if id == str(student_id):
            return {'name': name, 'id': id}
    return None

def edit_student(request, student_id):
    student = get_student_by_id(student_id)
    if not student:
        return redirect('dashboard')  # Or handle the case where student is not found

    if request.method == 'POST':
        new_name = request.POST.get('name')
        old_path = f'media/static/faces/{student["name"]}_{student["id"]}'
        new_path = f'media/static/faces/{new_name}_{student["id"]}'
        
        if os.path.exists(new_path):
            return render(request, 'attendance/edit_students.html', {'student': student, 'error': 'The new name directory already exists.'})

        try:
            os.rename(old_path, new_path)
        except Exception as e:
            return render(request, 'attendance/edit_students.html', {'student': student, 'error': str(e)})
        
        return redirect('dashboard')

    return render(request, 'attendance/edit_students.html', {'student': student})


def delete_student(request, student_id):
    student = get_student_by_id(student_id)
    if student:
        student_path = f'media/static/faces/{student["name"]}_{student["id"]}'
        if os.path.exists(student_path):
            shutil.rmtree(student_path)
        else:
            raise Http404(f"Student path {student_path} does not exist")
    else:
        raise Http404(f"Student with ID {student_id} not found")
    return redirect('dashboard')

def get_student_by_id(student_id):
    userlist = os.listdir('media/static/faces')
    print(f"Userlist: {userlist}")  # Debugging statement
    for user in userlist:
        try:
            name, id = user.rsplit('_', 1)
            print(f"Checking user: {user}, name: {name}, id: {id}")  # Debugging statement
            if id.lstrip('0') == str(student_id).lstrip('0'):  # Compare after stripping leading zeros
                return {'name': name, 'id': id}
        except ValueError:
            print(f"Skipping invalid filename: {user}")  # Debugging statement
            continue  # Skip if the filename does not match the expected pattern
    return None
