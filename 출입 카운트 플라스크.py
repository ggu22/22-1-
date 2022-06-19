import time
import cv2 
import numpy as np
from flask import Flask, render_template, Response
import Person

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')            #flask 서버에서 띄울 HTML 파일 (하위 폴더 /templates)


def gen():
#    cap = cv2.VideoCapture('http://192.168.0.27:8899/video')
#    cap = cv2.VideoCapture('http://192.168.0.71:81/stream')
    cap = cv2.VideoCapture('test.mp4')                              #비디오 가져오기 

    sub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)  #영상 전처리 
    
    kernelOp = np.ones((3, 3), np.uint8)      #영상 전처리 옵션 3개
    kernelOp2 = np.ones((5, 5), np.uint8)
    kernelCl = np.ones((11, 11), np.uint8)
    
    cnt_ent   = 0         #카운트 선언, 초기화
    cnt_exit = 0
    
    
    w = 640           #처리할 영상 해상도
    h = 960
    
    
    frameArea = h*w          #면적 계산용
    areaMin = frameArea/20   #최소값 카메라에 보이는 인물 크기에 따라 조절
    areaMax = frameArea/7     #최대값
    
    line_ent = int(3*(h/10))         #기준으로 삼을 좌표 조절
    line_exit   = int(7*(h/10))  
    ent_limit =   int(1*(h/5))
    exit_limit = int(4*(h/5))
           
    
    pt1 =  [0, line_exit];                       #라인 생성할 배열 생성
    pt2 =  [w, line_exit];
    pts_L1 = np.array([pt1, pt2], np.int32)
    pts_L1 = pts_L1.reshape((-1, 1, 2))
    pt3 =  [0, line_ent];
    pt4 =  [w, line_ent];
    pts_L2 = np.array([pt3, pt4], np.int32)
    pts_L2 = pts_L2.reshape((-1, 1, 2))

    pt5 =  [0, ent_limit];
    pt6 =  [w, ent_limit];
    pts_L3 = np.array([pt5, pt6], np.int32)
    pts_L3 = pts_L3.reshape((-1, 1, 2))
    pt7 =  [0, exit_limit];
    pt8 =  [w, exit_limit];
    pts_L4 = np.array([pt7, pt8], np.int32)
    pts_L4 = pts_L4.reshape((-1, 1, 2))
    

    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    persons = []
    max_p_age = 5
    pid = 1
    
    while(cap.isOpened()):
        ret, frame = cap.read()  
        for i in persons:
            i.age_one()
        
        try:
            image = cv2.resize(frame, (960, 640), None, 1, 1)       #영상 리사이즈 
            
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)       # 시계방향으로 90도         회전180도 회전  = cv2.ROTATE_180
            image = cv2.flip(image, -1)              # 좌우대칭 = -1 상하대칭 = 0  상하좌우대칭 = -1
            
    
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                #영상 전처리
            fgmask = sub.apply(gray)  
            
            
            retvalbin, bins = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)   #임계처리 바이너리 이미지 

            opening = cv2.morphologyEx(bins, cv2.MORPH_OPEN, kernelOp)      #모폴로지 연산 

            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernelCl)      #모폴로지 연산

            dilation = cv2.dilate(closing, kernelCl)        #이미지 팽창
            
            contours0, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   #바깥 경계선 찾고 라인 좌표 저장
  
        except:
            break
            

            
        for cnt in contours0:
                area = cv2.contourArea(cnt)          #객체의 면적 구하고 기준 초과하면 추적 시작
                if areaMax > area > areaMin:         #면적 기준에 맞는지 확인

                    M = cv2.moments(cnt)                  #면적 계산용 무게중심구하기
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    x,y,w,h = cv2.boundingRect(cnt)        #좌표를 직사각형으로   

                    new = True
                    if cy in range(ent_limit, exit_limit):            #라인 넘으면 ID부여 추적
                        for i in persons:
                            if abs(x-i.getX()) <= w and abs(y-i.getY()) <= h:
                                new = False
                                i.updateCoords(cx, cy)   
                                if i.going_UP(line_exit, line_ent) == True:            #라인 넘으면 카운트 1씩 증가
                                    cnt_ent += 1;
                                elif i.going_DOWN(line_exit, line_ent) == True:        #라인 넘으면 카운트 1씩 증가
                                    cnt_exit += 1;
                                break
                            if i.getState() == '1':
                                if i.getDir() == 'exit' and i.getY() > exit_limit:
                                    i.setDone()
                                elif i.getDir() == 'ent' and i.getY() < ent_limit:
                                    i.setDone()
                            if i.timedOut():
                                index = persons.index(i)  #persen.py 리스트에서 가져오기
                                persons.pop(index)
                                del i                     #메모리 해제
                        if new == True:
                            p = Person.MyPerson(pid, cx, cy, max_p_age)       #화면에 등장할 때 인식 , 객체 등록
                            persons.append(p)
                            pid += 1 
                            
                            

                    cv2.circle(image,(cx, cy), 5, (0, 0, 255), -1)                  #무게중심에 원 그리기 색상 B, G, R 순서
                    cv2.rectangle(image,(x, y),(x+w, y+h), (0, 255, 0), 2)            #탐지 박스 그리기


        
        for i in persons:
         
                cv2.putText(image, str(i.getId()), (i.getX(), i.getY()), font, 0.3, i.getRGB(), 1,cv2.LINE_AA)    #탐지되는 ID에 숫자 표시

        
        seat =     'SEAT:  ' + str('SET')                      #카운트 영상에 숫자 오버레이
        live =     'LIVE:   ' + str(cnt_ent - cnt_exit)
        str_ent =   'ENTER:   '+ str(cnt_ent)
        str_exit = 'EXIT: '+ str(cnt_exit)

        frame = cv2.polylines(image, [pts_L1], False, (255, 0, 0), thickness=2)    #지정한 배열에 맞춰 블루 라인 생성 thickness= 두께 색상 B, G, R 순서
        frame = cv2.polylines(image, [pts_L2], False, (0, 0, 255), thickness=2)    #레드 라인생성
        frame = cv2.polylines(image, [pts_L3], False, (255, 255, 255), thickness=1)    #하얀색 라인 생성
        frame = cv2.polylines(image, [pts_L4], False, (255, 255, 255), thickness=1)
      
        cv2.putText(image, seat, (10, 15), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)   #오버레이 할 숫자 좌표 크기 색 지정
        cv2.putText(image, seat, (10, 15), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)         #하얀 배경색으로 가시성 확보

        cv2.putText(image, live, (10, 35), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, live, (10, 35), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)    

        cv2.putText(image, str_ent, (150, 15), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str_ent, (150, 15), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(image, str_exit, (150, 35), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str_exit, (150, 35), font, 0.5, (255, 0, 0),1, cv2.LINE_AA)



        frame = cv2.imencode('.jpg', image)[1].tobytes()                             #영상 인코딩
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
      
        

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080) 
