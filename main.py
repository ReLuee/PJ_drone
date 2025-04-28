import os
import base64
import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import numpy as np

os.environ["STREAMLIT_WATCH_FILE"] = "false"

# YOLO 모델 로드
model = YOLO(r"C:\workspace\7_딥러닝\PJ_drone_save_human\runs\train\rps_yolov115\weights\best.pt")  # 사전 학습된 모델을 사용
# model.names[0]="person"

# 페이지 설정
st.set_page_config(page_title="드론으로 생명을 살리는 감지 시스템", page_icon="", layout="wide")

# 배경 설정 함수
def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# 배경 이미지 설정
# set_background("data/drone.png")

# 사이드바 메뉴
with st.sidebar:
    # st.image("data/logo.png", width=120)
    st.title("📑 드론 감지 시스템 목차")
    page = st.selectbox("이동할 섹션을 선택하세요:", ["홈", "소개", "사진 갤러리", "동영상 갤러리", "실시간 탐지", "문의하기"], key="sidebar_select")

# 새로고침 시 홈으로
if 'page' not in st.session_state:
    st.session_state.page = "홈"

if page != st.session_state.page:
    st.session_state.page = page
    st.rerun()

# 메인 페이지 제목
st.markdown(f"<h2 style='text-align:center;'>드론 감지 시스템</h2>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size:18px; font-weight:bold;'>📌 현재 페이지: {page}</div>", unsafe_allow_html=True)

# 공통 YOLO 탐지 함수
def detect_and_draw(image):
    results = model(image)
    result = results[0]
    boxes = result.boxes
    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()
    xyxy = boxes.xyxy.tolist()
    class_names = result.names
    labels = [class_names[int(i)] for i in class_ids]

    for i, box in enumerate(xyxy):
        if int(class_ids[i]) == 0:  # 0번 클래스: 사람
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'person {confidences[i]:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# 사진 갤러리 처리
if page == "사진 갤러리":
    st.title("📷 사진 갤러리")
    uploaded_image = st.file_uploader("📤 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        processed_img = detect_and_draw(open_cv_image)
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

# 동영상 갤러리 처리
elif page == "동영상 갤러리":
    st.title("🎞️ 동영상 갤러리")
    uploaded_video = st.file_uploader("📤 동영상을 업로드하세요", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())

        # 동영상 캡처
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()  # 실시간 프레임 표시용 placeholder

        frame_interval = 1 # 프레임 간격 줄이기 (더 빠르게 보여줌)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                processed = detect_and_draw(frame)
                stframe.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            frame_count += 1

        cap.release()

# 기타 페이지
elif page == "홈":
    st.title("🚀 드론을 활용한 생존자 유무 체크")
    st.write("이 시스템은 생존자 탐색을 위해 드론 데이터를 수집하고, 분석하여 시각화합니다.")

elif page == "소개":
    st.title("📘 소개")
    st.write("메타버스아카데미 대구 AI반 8팀 3차 프로젝트인 **폭격기**에서 개발한 드론 탐지 시스템입니다.")

elif page == "문의하기":
    st.title("📞 여기는 왜 눌러보셨나요. 문의할게 어딨다고? 문의할 내용 여기에.")
    st.write("이메일: BOOM@EXPLOSION.com")


# 추가//실시간 탐지
elif page == "실시간 탐지":
    st.title("실시간 촬영 및 탐지")
    
    
    if st.button("탐지 시작"):
        cap = cv2.VideoCapture(0)
        if st.button("탐지 종료"):
            cap.release()
            
        stframe = st.empty()
        
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed = detect_and_draw(frame)
            stframe.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    
