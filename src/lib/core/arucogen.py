import numpy as np
import cv2
import matplotlib.pyplot as plt
STANDARD_SIZE = 100
class ArUcoGenerator():
    def __init__(self,n):
        self._n = n
        self._write_obj_list = [np.zeros((STANDARD_SIZE, STANDARD_SIZE, 1), dtype="uint8") for i in range(n)]
        self._ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)        
    def run(self):
        print(f"Generating {self._n} ArUco markers...")
        n = self._n        
        Cols = 4           
        Rows = n // Cols 
        if n % Cols != 0:
            Rows += 1           
        Position = range(1,n + 1)
        fig = plt.figure(1)
        for i,obj in enumerate(self._write_obj_list):
            cv2.aruco.drawMarker(self._ARUCO_DICT, i, STANDARD_SIZE, obj, 1)            
            ax = fig.add_subplot(Rows,Cols,Position[i])
            ax.imshow(obj, cmap='gray')
            ax.axis('off')
        plt.show()        
        self._fig = fig
        
    def saveAll(self, folder_path):
        for i,obj in enumerate(self._write_obj_list):
            marker_path = f"{folder_path}/marker_{i}.jpeg"
            print(f"Saving in {marker_path} ...")
            cv2.imwrite(marker_path, obj)
        self._fig.savefig(f"{folder_path}/overall.jpeg")
