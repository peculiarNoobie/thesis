import os
import cv2
import mediapipe as mp
import math

mp_face_mesh=mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

for image in os.listdir('/home/gonzales/Thesis/BMI_Dataset02/Obese'):
        img=cv2.imread(os.path.join('/home/gonzales/Thesis/BMI_Dataset02/Obese', image))
        rgb_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = face_mesh.process(img)
        height, width, _ = img.shape

        for facial_landmarks in result.multi_face_landmarks:

                #CJWR

                f = open('/home/gonzales/Thesis/training_dataset/obese_dataset.csv', 'a')

                p1 = facial_landmarks.landmark[127] 
                xp1 = p1.x
                yp1 = p1.y

                p15 = facial_landmarks.landmark[264]
                xp15 = p15.x
                yp15 = p15.y

                p4 = facial_landmarks.landmark[58]
                xp4 = p4.x
                yp4 = p4.y

                p12 = facial_landmarks.landmark[367]
                xp12 = p12.x
                yp12 = p12.y

                p1p15_1 = math.sqrt((xp15 - xp1)**2 + (yp15 - yp1)**2)
                p4p12_1 = math.sqrt((xp12 - xp4)**2 + (yp12 - yp4)**2)
                cjwr = str(p1p15_1 / p4p12_1)

                #print ("CJWR", cjwr)

                #WHR
                p4 = facial_landmarks.landmark[58]
                xp4 = p4.x
                yp4 = p4.y

                p12 = facial_landmarks.landmark[367]
                xp12 = p12.x
                yp12 = p12.y

                p67 = facial_landmarks.landmark[13]
                xp67 = p67.x
                yp67 = p67.y

                n1 = facial_landmarks.landmark[8]
                xn1 = n1.x
                yn1 = n1.y

                p4p12_2 = math.sqrt((xp12 - xp4)**2 + (yp12 - yp4)**2)
                p67n1 = math.sqrt((xn1 - xp67)**2 + (yn1 - yp67)**2)
                whr = str(p4p12_2 / p67n1)

                #print ("WHR", whr)
                
                #PAR
                p1 = facial_landmarks.landmark[127]
                xp1 = p1.x
                yp1 = p1.y

                p4 = facial_landmarks.landmark[58]
                xp4 = p4.x
                yp4 = p4.y

                p8 = facial_landmarks.landmark[152]
                xp8 = p8.x
                yp8 = p8.y

                p12 = facial_landmarks.landmark[367]
                xp12 = p12.x
                yp12 = p12.y

                p15 = facial_landmarks.landmark[264]
                xp15 = p15.x
                yp15 = p15.y

                p67 = facial_landmarks.landmark[13]
                xp67 = p67.x
                yp67 = p67.y

                n5 = facial_landmarks.landmark[168]
                xn5 = n5.x
                yn5 = n5.y

                per1 = math.sqrt((xp4 - xp1)**2 + (yp4 - yp1)**2)
                per2 = math.sqrt((xp8 - xp4)**2 + (yp8 - yp4)**2)
                per3 = math.sqrt((xp12 - xp8)**2 + (yp12 - yp8)**2)
                per4 = math.sqrt((xp15 - xp12)**2 + (yp15 - yp12)**2)
                per5 = math.sqrt((xp1 - xp15)**2 + (yp1 - yp15)**2)
                perimeter = per1 + per2 + per3 + per4 + per5

                a = math.sqrt((xp15 - xp1)**2 + (yp15 - yp1)**2)
                b_1 = math.sqrt((xp12 - xp4)**2 + (yp12 - yp4)**2)
                h_1 = math.sqrt((xp67 - xn5)**2 + (yp67 - yn5)**2)
                a1 = ((a + b_1) / 2) * h_1

                h_2 = math.sqrt((xp8 - xp67)**2 + (yp8 - yp67)**2)
                b_2 = math.sqrt((xp12 - xp4)**2 + (yp12 - yp4)**2)
                a2 =   (h_2 * b_2) / 2

                area = a1 + a2

                par = str(perimeter / area)
                #print ("PAR", par)

                #ES
                p28 = facial_landmarks.landmark[33]
                xp28 = p28.x
                yp28 = p28.y

                p33 = facial_landmarks.landmark[359]
                xp33 = p33.x
                yp33 = p33.y

                p30 = facial_landmarks.landmark[112]
                xp30 = p30.x
                yp30 = p30.y

                p35 = facial_landmarks.landmark[362]
                xp35 = p35.x
                yp35 = p35.y

                p28p33 = math.sqrt((xp33 - xp28)**2 + (yp33 - yp28)**2)
                p30p35 = math.sqrt((xp35 - xp30)**2 + (yp35 - yp30)**2)
                es = str((p28p33 - p30p35) / 2)

                #print ("ES", es)

                #LF/HR
                n5 = facial_landmarks.landmark[168]
                xn5 = n5.x
                yn5 = n5.y

                p8 = facial_landmarks.landmark[152]
                xp8 = p8.x
                yp8 = p8.y

                n2 = facial_landmarks.landmark[10]
                xn2 = n2.x
                yn2 = n2.y

                lfh = math.sqrt((xp8 - xn5)**2 + (yp8 - yn5)**2)
                n2p8 = math.sqrt((xp8 - xn2)**2 + (yp8 - yn2)**2)
                lfhr = str(lfh / n2p8)

                #print ("LF/HR", lfhr)

                #FW/LFH
                p1 = facial_landmarks.landmark[127]
                xp1 = p1.x
                yp1 = p1.y

                p15 = facial_landmarks.landmark[264]
                xp15 = p15.x
                yp15 = p15.y

                n5 = facial_landmarks.landmark[168]
                xn5 = n5.x
                yn5 = n5.y

                p8 = facial_landmarks.landmark[152]
                xp8 = p8.x
                yp8 = p8.y

                p1p15_2 = math.sqrt((xp15 - xp1)**2 + (yp15 - yp1)**2)
                lfh = math.sqrt((xp8 - xn5)**2 + (yp8 - yn5)**2)
                fwlfh = str(p1p15_2 / lfh)

                #print ("FW/LFH", fwlfh)

                #MEH
                p22 = facial_landmarks.landmark[70]
                xp22 = p22.x
                yp22 = p22.y

                p28 = facial_landmarks.landmark[33]
                xp28 = p28.x
                yp28 = p28.y

                n3 = facial_landmarks.landmark[52]
                xn3 = n3.x
                yn3 = n3.y

                p29 = facial_landmarks.landmark[27]
                xp29 = p29.x
                yp29 = p29.y

                p25 = facial_landmarks.landmark[221]
                xp25 = p25.x
                yp25 = p25.y

                p30 = facial_landmarks.landmark[112]
                xp30 = p30.x
                yp30 = p30.y

                p19 = facial_landmarks.landmark[285]
                xp19 = p19.x
                yp19 = p19.y

                p35 = facial_landmarks.landmark[362]
                xp35 = p35.x
                yp35 = p35.y

                n4 = facial_landmarks.landmark[334]
                xn4 = n4.x
                yn4 = n4.y

                p34 = facial_landmarks.landmark[386]
                xp34 = p34.x
                yp34 = p34.y

                p16 = facial_landmarks.landmark[300]
                xp16 = p16.x
                yp16 = p16.y

                p33 = facial_landmarks.landmark[359]
                xp33 = p33.x
                yp33 = p33.y

                p22p28 = math.sqrt((xp28 - xp22)**2 + (yp28 - yp22)**2)
                n3p29 = math.sqrt((xp29 - xn3)**2 + (yp29 - yn3)**2)
                p25p30 = math.sqrt((xp30 - xp25)**2 + (yp30 - yp25)**2)
                p19p35 = math.sqrt((xp35 - xp19)**2 + (yp35 - yp19)**2)
                n4p34 = math.sqrt((xp34 - xn4)**2 + (yp34 - yn4)**2)
                p16p33 = math.sqrt((xp33 - xp16)**2 + (yp33 - yp16)**2)

                meh = str((p22p28 + n3p29 + p25p30 + p19p35 + n4p34 + p16p33) / 6)

                #print ("MEH", meh)

                f.write(cjwr + ', ')
                f.write(whr + ', ')
                f.write(par + ', ')
                f.write(es + ', ')
                f.write(lfhr + ', ')
                f.write(fwlfh + ', ')
                f.write(meh + '\n')

                f.close()

        cv2.waitKey(0)
        cv2.destroyAllWindows()
