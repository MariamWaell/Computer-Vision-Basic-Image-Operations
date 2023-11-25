import streamlit as st
import cv2
from filter import *
from histogram import *
from frequency import *
import seaborn as sns
from skimage import io
import matplotlib.pyplot as plt

#Layout
st.set_page_config(layout="wide")
tab1, tab2, tab3 = st.tabs(["Filters", "Histogram", "Frequency"])

#filters tab
with tab1:
    with st.container():
        col1,a,col2,b,col3,c,col4 = st.columns([2,0.2,2,0.2,2,0.2,2])
        with col1:
            image1= st.file_uploader ("Upload Image ", type= ["jpg","png","bmp","jpeg"], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        with col2:
            btn3= st.selectbox('Edge detection ',('Sobel', 'Roberts', 'Prewitt' ,'Canny'))
        with col3:
            btn1= st.selectbox('Add noise to image',('Uniform', 'Gaussian', 'salt & pepper'))
        with col4:
            btn2= st.selectbox('Filter Image',('Average', 'Gaussian', 'median'))
            btn_filter=st.selectbox("choose size",('3','5'))
                 
       
    with st.container():
        col1,a,col2,b,col3,c,col4 = st.columns([2,0.2,2,0.2,2,0.2,2]) 
        if image1:
            path = "images/" + image1.name
            image= cv2.imread(path,cv2.IMREAD_GRAYSCALE) 
            with col1:
                st.image(image, caption='Original image' ,width=300) 
                    
            with col2:
                # Blur the image for better edge detection
                img_blur = cv2.GaussianBlur(image, (3,3), 0)
                if(btn3=='Sobel'):
                    img_edge_detected=SobelXY(img_blur)
                    st.image(img_edge_detected, caption='edge detected image' ,width=300,clamp=True)  
                elif(btn3=='Roberts'):
                    img_edge_detected=Robert(image)
                    st.image(img_edge_detected, caption='edge detected image' ,width=300,clamp=True)   
                elif(btn3=='Prewitt'):
                    img_edge_detected=prewitt(img_blur)
                    st.image(img_edge_detected, caption='edge detected image' ,width=300,clamp=True) 
                elif(btn3=='Canny'):
                    img_edge_detected=cannyDetection(img_blur)
                    st.image(img_edge_detected, caption='edge detected image' ,width=300,clamp=True) 
                                  
            with col3:    
                image= cv2.imread(path,cv2.IMREAD_GRAYSCALE) 
                if(btn1=='Uniform'):
                    img_noised=Addnoise(image,'Uniform')
                    #st.image(img_noised, caption='noised image' ,width=300,clamp=True) 
                    st.image("images\\result_image.bmp", caption='noised image' ,width=300)
                elif(btn1=='Gaussian'):
                    img_noised=Addnoise(image,'Gaussian')
                    #st.image(img_noised, caption='noised image' ,width=300,clamp=True) 
                    st.image("images\\noisy.bmp", caption='noised image' ,width=300)
                elif(btn1=='salt & pepper'):
                    img_noised=Addnoise(image,'salt & pepper')
                    st.image(img_noised, caption='noised image' ,width=300,clamp=True)  
          
            with col4:
                if(btn1=='Uniform'): 
                    noise=cv2.imread("images\\result_image.bmp",cv2.IMREAD_GRAYSCALE)
                elif(btn1=='Gaussian'): 
                    noise=cv2.imread("images\\noisy.bmp",cv2.IMREAD_GRAYSCALE)
                elif(btn1=='salt & pepper'):
                    noise = img_noised
                    
                if(btn2=='Average'):
                    if(btn_filter=='3'):
                        img_filtered=MeanFilter(noise, 9)
                        st.image(img_filtered, caption='filtered image' ,width=300,clamp=True) 
                    elif(btn_filter=='5'):
                        img_filtered=MeanFilter(noise, 25)
                        st.image(img_filtered, caption='filtered image' ,width=300,clamp=True) 
                    
                elif(btn2=='Gaussian'):
                    if(btn_filter=='3'):
                        img_filtered=gaussian_filter(noise, 3, sigma=1)
                        st.image(img_filtered, caption='filtered image' ,width=300,clamp=True)
                    elif(btn_filter=='5'):
                        img_filtered=gaussian_filter(noise, 5, sigma=1)
                        st.image(img_filtered, caption='filtered image' ,width=300,clamp=True)
                  
                elif(btn2=='median'):
                    if(btn_filter=='3'):
                        img_filtered=median_filter(noise, 3)
                        st.image(img_filtered, caption='filtered image' ,width=300,clamp=True) 
                    elif(btn_filter=='5'):    
                        img_filtered=median_filter(noise, 5)
                        st.image(img_filtered, caption='filtered image' ,width=300,clamp=True) 
                    
                                                 
#histogram tab                              
with tab2:
    with st.container():
        col1,col2,col3 = st.columns([2,2,2])
        with col1:
            img2= st.file_uploader ("Upload Image", type= ["jpg","png","bmp","jpeg"], accept_multiple_files=False, key=2, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        with col3:
            btn3=st.selectbox("Thresholding",options=('local','global'))
        with col2:
            btn4=st.selectbox("Processing",options=('Equalization','Normalization')) 

    with st.container():
        col1,col2,col3 = st.columns([2,2,2])
        with col1:
           if img2:
                path1 = "images/" + img2.name
                im2=cv2.imread(path1,cv2.IMREAD_GRAYSCALE) 
                st.image(im2, caption='Original image' ,width=300)
                 
        with col2:
            if(img2):
                path1 = "images/" + img2.name
                im2=cv2.imread(path1,cv2.IMREAD_GRAYSCALE) 
                if(btn4=='Equalization'):
                    equalized_image=equalization(im2,256)
                    st.image(equalized_image, caption='Equalized image' ,width=300) 
                if(btn4=='Normalization'):
                    normalize(img=im2)
                    st.image("images\\normalized.bmp", caption='Normalized image' ,width=300)    
                    
        with col3:       
            if(img2):
                if(btn3=='local'):
                    Local=local_thresholding(im2,60,60,60,60)
                    st.image(Local, caption='Local Threshold image' ,width=300) 
                elif(btn3=='global'):
                    Global= global_thresholding(im2,120)
                    st.image(Global, caption='Global Thresholding image' ,width=300)


    with st.container():
        col1,col2 = st.columns([2,2])
        if(img2):
            with col1:
                image= cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
                sns.distplot(image, color="grey", label="Density")
                plt.title('Grayscale Histogram')
                plt.xlabel('Intensity Value')
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
            with col2:
                im= io.imread(path1)
                RGB_histogram = plt.hist(im[:, :, 0].ravel(), bins = 256, color = 'red')
                RGB_histogram = plt.hist(im[:, :, 1].ravel(), bins = 256, color = 'Green')
                RGB_histogram = plt.hist(im[:, :, 2].ravel(), bins = 256, color = 'Blue')
                plt.title('RGB Histograms')
                RGB_histogram = plt.xlabel('Intensity Value')
                RGB_histogram = plt.ylabel('Count')
                RGB_histogram = plt.legend([ 'Red_Channel', 'Green_Channel', 'Blue_Channel'])   
                st.pyplot()     
        
                
    with st.container():
        col1,col2,col3=st.columns([2,2,2])
        if(img2):
            im= io.imread(path1)
            with col1:
                R=im[:, :, 0]
                R_Distribution = sns.distplot(R, color="red", label="Compact")
                R_Distribution= plt.xlabel('Intensity Value')
                plt.title('R Distribution')
                st.pyplot()
                R_Cumulative = plt.hist(R.ravel(), bins = 256, cumulative = True,color="red")
                R_Cumulative= plt.xlabel('Intensity Value')
                R_Cumulative= plt.ylabel('Count') 
                plt.title('R Cumulative')
                st.pyplot()
            with col2:    
                G=im[:, :, 1]
                G_Distribution= sns.distplot(G, color="green", label="Compact")
                G_Distribution= plt.xlabel('Intensity Value')
                plt.title('G Distribution')
                st.pyplot()
                G_Cumulative= plt.hist(G.ravel(), bins = 256, cumulative = True,color="green")
                G_Cumulative= plt.xlabel('Intensity Value')
                G_Cumulative= plt.ylabel('Count') 
                plt.title('G Cumulative')
                st.pyplot()
            with col3:
                B=im[:, :, 2]
                B_Distribution = sns.distplot(B, color="blue", label="Compact")
                B_Distribution= plt.xlabel('Intensity Value')
                plt.title('B Distribution')
                st.pyplot()
                B_Cumulative = plt.hist(B.ravel(), bins = 256, cumulative = True,color="blue")
                B_Cumulative= plt.xlabel('Intensity Value')
                B_Cumulative= plt.ylabel('Count') 
                plt.title('B Cumulative')
                st.pyplot()
            
        
#hybrid image tab
with tab3:
  with st.container():
    col1,c,col2,c,col3 = st.columns([2,0.3,2,0.2,1.5])
    with col1:
        img1= st.file_uploader ("Upload Image 1", type= ["jpg","png","bmp","jpeg"], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    with col2:
        img2= st.file_uploader ("Upload Image 2", type= ["jpg","png","bmp","jpeg"], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    with col3:
        x= st.button("Generate hybrid image") 
            
            
    with st.container():
        col1,col2, col3 = st.columns([2,2,2])
        with col1:
           if img1:
                path1 = "images/" + img1.name
                image1 = cv2.imread(path1,0) 
                image1 = image1.astype(np.float32)/255
                lowpass =LowPass(image1)
                st.image(lowpass, caption='Low Pass Filtered image' ,width=350,clamp=True) 
        with col2:
             if img2:
                path2 = "images/" + img2.name
                image2 = cv2.imread(path2,0) 
                image2 = image2.astype(np.float32)/255
                highpass=HighPass(image2)
                st.image(highpass, caption='High Pass Filtered image' ,width=350,clamp=True) 
        with col3:
            if (img1 and img2 and x):
                hy_img=hybrid(lowpass,highpass)
                st.image(hy_img, caption='Hybrid image' ,width=350,clamp=True)
            
css = '''
    <style>
    section.main > div:has(~ footer ) {
        padding-bottom: 5px;

        padding-left:15px;
    }
    </style>
    '''
st.markdown(css, unsafe_allow_html=True)   
