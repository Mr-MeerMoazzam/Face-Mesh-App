import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import time
import tempfile
from PIL import Image

DEMO_IMAGE= 'resources/demo.jpg'
DEMO_VIDEO= 'resources/demo.mp4'

mp_drawing=mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

st.title('Face Mesh App Using MediaPipe')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false] > div:first-child{
    width: 350px
    margin-left: -350px
}
    </style>

    """,
    unsafe_allow_html=True,
)
st.sidebar.title('FaceMesh Sidebar')
st.sidebar.subheader('Parameters')

@st.cache_data()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

app_mode= st.sidebar.selectbox('Choose the App mode ',
                               ['About App','Run on Image','Run on Video'])
if app_mode=='About App':
    st.markdown('In this application we are going to use **Media Pipe** for Creating a Face Mesh App. **Streamlit** is to create a interective GUI. ')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false] > div:first-child{
    width: 350px
    margin-left: -350px
}
    </style>

    """,
    unsafe_allow_html=True,
)
    # st.video('Paste vidoe link here')
    st.markdown('''
                # About Me
                
                My name is Moazzam Ali. I am an Associate Machine Learning Engineer, Python Developer, Data Analyst, and Proactive AI (Artificial Intelligence) student at Air University Islamabad with substantial academic achievements and volunteering experience. I have a strong mind for research work, and I am currently working to understand better how machine learning algorithms can be more beneficial for humanity. My expertise includes  Python, ML algorithm selection, cross-validation, digital image processing, Computer Vision, feature engineering, data analysis and interpretation, and problem-solving. I enjoy generating new ideas and devising feasible solutions to broadly relevant problems will allow me to develop and promote technologies that be. My colleagues would describe me as a driven, resourceful individual who maintains a positive, proactive attitude when faced with adversity. Currently, I’m seeking opportunities in specific fields of interest. Specific fields of interest include data analytics, machine learning, computer vision, and Natural Language Processing.
                Also check me out on Social Media
                - [Youtube](https://youtube.com/@mtlearners6563)
                - [LinkedIn](https://www.linkedin.com/in/meermoazzam/)
                ''')
elif app_mode =='Run on Image':
    drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false] > div:first-child{
    width: 350px
    margin-left: -350px
}
    </style>

    """,
    unsafe_allow_html=True,
)
    st.markdown("**detected Faces**")
    kpi1_text = st.markdown("0")
    max_faces=st.sidebar.number_input('Maximum Number of Faces', value=2,min_value=1)
    
    st.sidebar.markdown('---')
    detection_confidence= st.sidebar.slider('Minimum Detection Confidence', min_value=0.0,max_value=1.0,value=0.5)
    st.sidebar.markdown('---')
    img_file_buffer=st.sidebar.file_uploader('Upload an image',type=['jpg','jpeg','png'])
    
    if img_file_buffer is not None:
        image=np.array(Image.open(img_file_buffer))
    else:
        demo_image= DEMO_IMAGE
        image=np.array(Image.open(demo_image))
    
    st.sidebar.text("Original Image")
    st.sidebar.image(image)
    face_count=0
    ## Dashboard
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_faces,
        min_detection_confidence=detection_confidence
    ) as face_mesh:
        results=face_mesh.process(image)
        out_image=image.copy()
        ##FAce landmark Drawing 
        for face_landmarks in results.multi_face_landmarks:
            face_count += 1
            mp_drawing.draw_landmarks(
                image=out_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec
                )
        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(out_image,use_column_width=True)
elif app_mode =='Run on Video':
    
    st.set_option('deprecation.showfileUploaderEncoding', False)    
    use_webcam=st.sidebar.button('Use Webcam')
    record= st.sidebar.checkbox('Record Video')
    
    if record:
        st.checkbox("Recording",value=True)
        
    drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    st.sidebar.markdown('---')
    st.markdown(
    
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false] > div:first-child{
    width: 350px
    margin-left: -350px
}
    </style>

    """,
    unsafe_allow_html=True,
)
    max_faces=st.sidebar.number_input('Maximum Number of Faces', value=5,min_value=1)
    st.sidebar.markdown('---')
    detection_confidence= st.sidebar.slider('Minimum Detection Confidence', min_value=0.0,max_value=1.0,value=0.5)
    st.sidebar.markdown('---')
    tracking_confidence= st.sidebar.slider('Minimum Tracking Confidence', min_value=0.0,max_value=1.0,value=0.5)
    st.sidebar.markdown('## Output')
    
    stframe=st.empty()
    video_file_buffer=st.sidebar.file_uploader('Upload a Video',type=['mp4','mov','avi'])
    
    tfile=tempfile.NamedTemporaryFile(delete=False)
    
    ## We will get our input video here
    if not video_file_buffer:
        if use_webcam:
            vid=cv2.VideoCapture(0)
        else:
            vid=cv2.VideoCapture(DEMO_VIDEO)
            tfile.name=DEMO_VIDEO
    else:
        tfile.write(video_file_buffer.read())
        vid=cv2.VideoCapture(tfile.name)
    
    width=int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input= int(vid.get(cv2.CAP_PROP_FPS))
    
    ## Recording Part
    
    codec=cv2.VideoWriter_fourcc('M','J','P','G')
    out=cv2.VideoWriter('output1.mp4',codec,fps_input,(width,height))
    
    
    st.sidebar.text('Input Video')
    st.sidebar.video(tfile.name)

    fps=0
    i=0
    
    drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    
    kpi1,kpi2,kpi3=st.columns(3)
    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text=st.markdown("0")
    with kpi2:
        st.markdown("**Face Detected**")
        kpi2_text=st.markdown("0")
    with kpi3:
        st.markdown("**Frame Width**")
        kpi3_text=st.markdown("0")
        
    st.markdown("<hr/>",unsafe_allow_html=True)
    
    
    ## Face Mesh Predictor
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=max_faces,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence
        
    ) as face_mesh:
        prevTime=0
        while vid.isOpened():
            ret, frame=vid.read()
            if not ret:
                continue
            
            frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results=face_mesh.process(frame)
            frame.flags.writeable=True
            frame= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            face_count=0
            if results.multi_face_landmarks:
                 ##FAce landmark Drawing 
                for face_landmarks in results.multi_face_landmarks:
                    face_count += 1
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                        )
            #FPS Counter logic
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            
            if record:
                out.write(frame)

            #Dashboard 
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)
            
            frame=cv2.resize(frame,(0,0),fx=0.8,fy=0.8)
            frame=image_resize(image=frame,width=640)
            stframe.image(frame,channels='BGR', use_column_width=True)
    st.text('Video Processed')

    output_video = open('output1.mp4','rb')
    out_bytes = output_video.read()
    st.video(out_bytes)
    vid.release()
    out. release()
        
            