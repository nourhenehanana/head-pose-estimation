import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)

mp_drawing=mp.solutions.drawing_utils
drawing_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=1)


cap=cv2.VideoCapture(0)
start=0
while cap.isOpened():
    succss,image=cap.read()
    start=time.time()
    #flip the image horizontally for a later selfie view display
    #convert the color space from bgr to rgb 
    image=cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)
    #to improve performance
    image.flags.writeable=False
    #get the result
    results=face_mesh.process(image)
    #to improve performance 
    image.flags.writeable=True 
    #convert the color space from rgb to bgr
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    img_h,img_w,img_c=image.shape
    face_3d=[]
    face_2d=[]
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
            cv2.putText(image,text,(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            cv2.putText(image,"x:"+str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            cv2.putText(image,"y:"+str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            cv2.putText(image,"z:"+str(np.round(z,2)),(500,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
        
       
        
        mp_drawing.draw_landmarks(image=image,
                                  landmark_list=face_landmarks,
                                  connections=mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=drawing_spec,
                                  connection_drawing_spec=drawing_spec)
        
            
            
                
    
    
    cv2.imshow("Image",image)
    key = cv2.waitKey(1) & 0xFF
    if key==ord("q"):
        break

cv2.destroyAllWindows()