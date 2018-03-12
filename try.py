import numpy as np
import cv2
from matplotlib import pyplot as plt

def main():
    
    #cv2.waitKey(0)
    #drawing
    #camaraWeb()    
    #matplotlib()
    #blending()
    #ReadPixels()
    #Layers
    #Region()
    #JoinImages()

    return 

def ReadPixels():

    image=cv2.imread("Images/BugsBunny.png")

    px=image[420,500]

    image[100,100]=[255,255,255]

    print (px)

    print (image.shape)

    print (image.size)

    print (image.dtype)

    cv2.imshow("Image window",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return
    
def JoinImages():
    
    #Load Images
    img1 = cv2.imread('Images/goal.jpg')
    img2 = cv2.imread('Images/Figure2.jpg')

    #Obtain a region of the img1 of the shape of img2
    filas,cols,canales = img2.shape
    #Stablish the are taked from the img1
    area = img1[0:filas, 0:cols ]

    #Extract the figure of the logo without colors (GrayScale)
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #A threshold is set to eliminate the background or contour. The result is saved into a "mask" 
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    #We take the inverse part of the mask. Here we obtain the dark part.
    mask_inv = cv2.bitwise_not(mask)

    #Invert colors. With this we put the colors(Not black)
    img1_bg = cv2.bitwise_and(area,area,mask = mask_inv)

    #Obtain the colors of the original image. Extract the original colors from the second image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    #Adding the changes to the general image. Apply the changes to the image. Here we have the both images joined.
    dst = cv2.add(img1_bg,img2_fg)

    #Fill the first image with the changes saved in dst.
    img1[0:filas, 0:cols ] = dst

    #Show the window image
    cv2.imshow('res',img1)

    #Wait util a key is preseed
    cv2.waitKey(0)
    
    #Destroy the window when the key was pressed
    cv2.destroyAllWindows()

    return

def Layers():

    #Layer split
    img = cv2.imread('img/BugsBunny')

    #Split image in the layers r,g,b
    b,g,r=cv2.split(img)

    #Merge or union of layers
    img = cv2.merge((b,g,r))

    #show images
    cv2.imshow('Layers',img)

    #wait key to close window
    cv2.waitKey(0)

    #Destroy window when the key is press
    cv2.destroyAllWindows()   

    return

def Region():

    img = cv2.imread('Images/BugsBunny.png')

    #Extract Region
    region = img[280:340, 330:390]

    #Fill Region of image with the new region
    img[10:70, 100:160] = region

    #show images
    cv2.imshow('Imagen',img)
    #wait key to close window
    cv2.waitKey(0)
    #Destroy window when the key is press
    cv2.destroyAllWindows()   

    return

def blending():
    #Load images
    img1 = cv2.imread('images/BugsBunny.png')
    img2 = cv2.imread('images/LolaBunny.png')

    #Merge 
    dst = cv2.addWeighted(img2,0.7,img1,0.3,0)
    #show images
    cv2.imshow('dst',dst)
    #wait key to close window
    cv2.waitKey(0)
    #Destroy window when the key is press
    cv2.destroyAllWindows()    

    return

def drawing():

    #Set img whit the size
    img = np.zeros((512,512,3),np.uint8)

    #Draw Primitives
    img = cv2.line(img,(0,0),(511,511),(255,0,0),5)
    img = cv2.circle(img,(255,255),65,(0,255,0),1)

    #It is possible draw polygons. It's neccessary to have an array of points.
    pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
    pts = pts.reshape((-1,1,2))
    img = cv2.polylines(img,[pts],True,(255,255,255))
    cv2.imshow('Imagen',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def matplotlib():

    #img=cv2.imread("Images/goal.jpg")
    #plt.imshow(img)
    #plt.xticks([]),plt.yticks([])
    #plt.show()

    return

def camaraWeb():

    cap = cv2.VideoCapture(0)

    while(True):
        #Captura frame per frame
        ret, frame = cap.read()
        
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF== ord('q'):
            break

        #when everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return

if __name__=='__main__':
    main()