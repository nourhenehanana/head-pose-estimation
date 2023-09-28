import cv2
import mediapipe as mp
import numpy as np
import time
import random

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
start_time = time.time()
timer_duration = 4  # Timer duration in seconds
action_duration = 4  # Duration for which the action is displayed in seconds

# List of available texts
texts = ["looking left", "looking right", "looking down", "looking up", "forward"]

def generate_random_action(previous_actions):
    action = random.choice(texts)
    while action in previous_actions:
        action = random.choice(texts)
    return action

current_action = generate_random_action([])
next_action_time = start_time + timer_duration
show_action_until = start_time + action_duration

correct_actions = 0
previous_actions = []

while cap.isOpened():
    success, image = cap.read()
    current_time = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx,lm in enumerate(face_landmarks.landmark):
                if idx==33 or idx==263 or idx==1 or idx==61 or idx==291 or idx==199:
                    if idx==1:
                        nose_2d=(lm.x*img_w,lm.y*img_h)
                        nose_3d=(lm.x*img_w,lm.y*img_h,lm.z*3000)
                    x,y=int(lm.x*img_w),int(lm.y*img_h)
                    #get the 2D coordinates
                    face_2d.append([x,y])
                    #get the 3D coordinates 
                    face_3d.append([x,y,lm.z])
            #convert to the numpy array
            face_2d=np.array(face_2d,dtype=np.float64)
            face_3d=np.array(face_3d,dtype=np.float64)
            #the camera matrix
            focal_length=1*img_w
            cam_matrix=np.array([[focal_length,0,img_h/2],[0,focal_length,img_w/2],
                                [0,0,1]])
            #the distortion parameters
            dist_matrix=np.zeros((4,1),dtype=np.float64)
            #solve pnp
            success,rot_vec,trans_vec=cv2.solvePnP(face_3d,face_2d, cam_matrix,dist_matrix)
            #success is that the system has successfully estimated the pose
            #rot_vec indicates how much the points are rotates
            #trans_vec how much the points are translated around
            
            #get rotational matrix rmat and jac is the jacobian matrix
            rmat,jac=cv2.Rodrigues(rot_vec)
            
            #get angles
            angles,mtxR,mtxQ,Qx,Qy,Qz=cv2.RQDecomp3x3(rmat)
            
            #get the y rotation degree
            x=angles[0]*360
            y=angles[1]*360
            z=angles[2]*360
            
            #See where the user's head titling
            if y<-10:
                text="looking left"
            elif y>10:
                text="looking right"
            elif x<-10:
                text="looking down"
            elif x>10:
                text="looking up"
            else:
                text="forward"
           
            #Display the nose direction
            nose_3d_projection,jacobian=cv2.projectPoints(nose_3d,rot_vec,trans_vec,cam_matrix,dist_matrix)
            
            p1=(int(nose_2d[0]),int(nose_2d[1]))
            p2=(int(nose_2d[0]+y*10),int(nose_2d[1]-x*10))
            
            cv2.line(image,p1,p2,(255,0,0),3)
            
            #Add the text on the image
            #cv2.putText(image,text,(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            print(text)
            cv2.putText(image,"x:"+str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            cv2.putText(image,"y:"+str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            cv2.putText(image,"z:"+str(np.round(z,2)),(500,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
        
       
        
        mp_drawing.draw_landmarks(image=image,
                                  landmark_list=face_landmarks,
                                  connections=mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=drawing_spec,
                                  connection_drawing_spec=drawing_spec)
   
    if current_time < next_action_time and current_time<= show_action_until:
        cv2.putText(image, current_action, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Check if the user performed the correct action
        if current_action==text:
            correct_actions += 1
            previous_actions.append(current_action)
            if correct_actions >= 3:
                print("Congratulations! You completed the game!")
                break
            current_action = generate_random_action(previous_actions)
            next_action_time = current_time + timer_duration
            show_action_until = current_time + action_duration
    elif current_time >= next_action_time:
        print("The process failed.")
        break
    #else:
        #current_action = generate_random_action(previous_actions)
        #next_action_time = current_time + timer_duration
        #show_action_until = current_time + action_duration

    cv2.imshow("Image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
