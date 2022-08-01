from tkinter import Label, filedialog
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    #return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    
    draw_lines(line_image, lines,(255,0,0),10)
    
    return line_image 

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)
    # return cv2.addWeighted(img, alpha=0.8, src2=initial_img, beta=1., gamma=0.)


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[4]:


import os
os.listdir("test_images/")


# In[5]:


cwd = os.getcwd()

print(cwd)

cwdsplit = cwd.split("\\")

print(cwdsplit)

#cwd_edited = cwdsplit[0] + 

cwd_edited =""

for i in range(len(cwdsplit)):
    cwd_edited += cwdsplit[i] + "/"
    
print(cwd_edited)    


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[6]:


test_images = []

test_image_names= []

for image_unread in os.listdir(cwd_edited+"test_images/"):
    test_image_names.append(image_unread)
    image_read = mpimg.imread(cwd_edited+"test_images/"+image_unread)
    test_images.append(image_read)

print("Image Names List :" + str(test_image_names))  

print("\n")
    
print("Images Data :"+str(test_images))


# In[7]:


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

###PARAMATERS USED###

#GAUSSIAN BLUR PARAMETER
kernel_size =5
#CANNY EDGE DETECTION PARAMETERS
low_threshold = 50
high_threshold = 150
# HOUGH LINE TRANSFORM PARAMETERS
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 45   # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40 #minimum number of pixels making up a line
max_line_gap = 100    # maximum gap in pixels between connectable line segments


# In[8]:


def Averaged_LeftRight_Lanes(Houghlines):
    leftlane_lines =[]
    leftlane_weights = []
    rightlane_lines = []
    rightlane_weights = []
    
    
    for line in Houghlines:
        for x1,y1,x2,y2 in line:
            #for non-vertical lines
            if (x1!=x2):
                slope = (y2-y1)/(x2-x1)
                intercept= y1 - slope*x1
                linelength = np.sqrt((y2-y1)**2 + (x2-x1)**2)
                if(slope<0):
                    leftlane_lines.append((slope,intercept))
                    leftlane_weights.append((linelength))
                elif(slope>0):
                    rightlane_lines.append((slope,intercept))
                    rightlane_weights.append((linelength))
                    
                    
    #Calculation of the averaged Left Lane Line and Right Lane Line
   
    LeftLane = (lambda:None, lambda:np.dot(leftlane_weights,leftlane_lines)/np.sum(leftlane_weights)) [len(leftlane_weights)>0]()
    
    #LeftLane = np.dot(leftlane_weights,leftlane_lines)/np.sum(leftlane_weights)
    
    RightLane = (lambda:None, lambda:np.dot(rightlane_weights,rightlane_lines)/np.sum(rightlane_weights)) [len(rightlane_weights)>0]()
    
    #RightLane = np.dot(rightlane_weights,rightlane_lines)/np.sum(rightlane_weights)
    
    
    return LeftLane,RightLane 


# In[9]:


def getCoords_LeftRight_Lanes(line,image):
    
    bottom_ycoord = image.shape[0]
    
    top_ycoord = bottom_ycoord*0.6
    
    slope,intercept = line
        
    x1 = int((bottom_ycoord - intercept)/slope)
    
    x2 = int((top_ycoord - intercept)/slope)
    
    y1 = int(bottom_ycoord)
    
    y2 = int(top_ycoord)
    
    return [(x1,y1,x2,y2)]


def Final_LeftRight_Lanes(image,Houghlines):
    
    LeftLane,RightLane = Averaged_LeftRight_Lanes(Houghlines)
    
    Left_Line_Coords = getCoords_LeftRight_Lanes(LeftLane,image)
    
    Right_Line_Coords = getCoords_LeftRight_Lanes(RightLane,image)
    
    return Left_Line_Coords , Right_Line_Coords   


# In[10]:


def Draw_Averaged_Extrapolated_LeftRight_Lanes(line_image,masked_edges,original_image, rho, theta, threshold,min_line_length, max_line_gap):
    
    houghlines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    draw_lines(line_image,Final_LeftRight_Lanes(original_image,houghlines) ,(255,0,0),10)
       
    return line_image  


# In[11]:


def LaneLines_Detector(image,image_name,index):    
    grayimg = grayscale(image)
    
    gaussianblur_grayimg = gaussian_blur(grayimg,kernel_size)
    
    edges = canny(gaussianblur_grayimg,low_threshold,high_threshold)
    
    imshape = edges.shape
    
    vertices = np.array([[(0,imshape[0]),(465, 320), (475, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    
    masked_edges = region_of_interest(edges,vertices)
    
    masked_edges_image = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
        
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    #hough_image_colored = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)
    
    color_edges = np.dstack((edges, edges, edges)) 
    
    #line_image = np.zeros((masked_edges.shape[0], masked_edges.shape[1], 3), dtype=np.uint8)\
    
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    
    #lines_edges = weighted_img(line_image,color_edges, 0.8, 1, 0)    
    
    #lines_edges = weighted_img(color_edges,line_image, 0.8, 1, 0)   
    
    
    
    lines_edges = weighted_img(lines,image, 0.8, 1, 0)  
    
    ExtrapolatedLines_Image = Draw_Averaged_Extrapolated_LeftRight_Lanes(line_image,masked_edges,image, rho, theta, threshold,min_line_length, max_line_gap)
    
    Extrapolated_Avged_Lines_Edges = weighted_img(ExtrapolatedLines_Image,image, 0.8, 1, 0)
    
    
    f, ax = plt.subplots(3, 2, figsize=(11,11))
           
    f.suptitle("Graphs of the Image : " + image_name)
    
    ax[0,0].set_title('Original Image')
    ax[0,0].imshow(image)

    ax[0,1].set_title('GreyScaled Image')
    ax[0,1].imshow(grayimg, cmap='gray')
   
    ax[1,0].set_title('Image with Masked Edges')
    ax[1,0].imshow(masked_edges_image)

    ax[1,1].set_title('Hough Transformed Image')
    ax[1,1].imshow(lines)

    ax[2,0].set_title('Final Image with Raw Lane Lines')
    ax[2,0].imshow(lines_edges)  
    
    ax[2,1].set_title('Final Image with Extrapolated and Averaged Lane Lines')
    ax[2,1].imshow(Extrapolated_Avged_Lines_Edges)


# In[12]:


for index in range(len(test_images)):
    LaneLines_Detector(test_images[index],test_image_names[index],index)


# In[13]:


def Video_LaneLines_Detector(image):
    
    grayimg = grayscale(image)
    
    gaussianblur_grayimg = gaussian_blur(grayimg,kernel_size)
    
    edges = canny(gaussianblur_grayimg,low_threshold,high_threshold)
    
    imshape = edges.shape
    
    vertices = np.array([[(0,imshape[0]),(465, 320), (475, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    
    masked_edges = region_of_interest(edges,vertices)
    
    masked_edges_image = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
        
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    color_edges = np.dstack((edges, edges, edges)) 
    
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    
    #lines_edges = weighted_img(line_image,color_edges, 0.8, 1, 0)    
    
    #lines_edges = weighted_img(color_edges,line_image, 0.8, 1, 0)
    
    #print(lines.shape)
    
    #print(image.shape)
    
    #if(lines.shape != image.shape):
        #lines = cv2.resize(lines, (image.shape[0], image.shape[1])) 
        
        
    #if(lines.shape != image.shape):
         #image = cv2.resize(image, (lines.shape[1], lines.shape[0]))
            
    #print(lines.shape)
    
    #print(image.shape)        
    
    lines_edges = weighted_img(lines,image, 0.8, 1, 0)  
    
    #if(lines.shape != image.shape):
        #lines_edges = weighted_img(lines,grayimg, 0.8, 1, 0)  
    #elif(lines.shape == image.shape):
        #lines_edges = weighted_img(lines,image, 0.8, 1, 0)  
          
    ExtrapolatedLines_Image = Draw_Averaged_Extrapolated_LeftRight_Lanes(line_image,masked_edges,image, rho, theta, threshold,min_line_length, max_line_gap)
    
    Extrapolated_Avged_Lines_Edges = weighted_img(ExtrapolatedLines_Image,image, 0.8, 1, 0)
    
    
    return Extrapolated_Avged_Lines_Edges
    

from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[15]:


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = Video_LaneLines_Detector(image)
    return result

filename = ''

def select_file():
    filetypes = (
        ('video files', '*.mp4'),
        ('All files', '*.*')
    )
    global filename
    filename = fd.askopenfilename(
        title='Select video',
        initialdir='./',
        filetypes=filetypes)

    print('Selected:', filename)
    do_process()

    open_file()

def open_file():
    cap = cv2.VideoCapture("output/output.mp4")
    if (cap.isOpened()== False): 
        print("Error opening video  file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
    cap.release()
    cv2.destroyAllWindows()


def do_process():
    global filename
    white_output = 'output/output.mp4'
    clip1 = VideoFileClip(filename)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')
    HTML("""
    <video width="960" height="540" controls>
    <source src="{0}">
    </video>
    """.format(white_output))


# create the root window
root = tk.Tk()
root.title('Lane Line Detection')
root.geometry('400x300')

label = Label(root,text="Lane Line Detection")
label.pack(pady=20)

# open button
open_button = ttk.Button(
    root,
    text='Select Video',
    command=select_file
)


open_button.pack(expand=True)

root.mainloop()


