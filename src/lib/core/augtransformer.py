import numpy as np
import cv2


class AugTransformer():
    def __init__(self, source):
        self.source = source
        self.nothing = None

    def transform(self, image):
        (imgH, imgW) = image.shape[:2]
        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
        if len(corners) != 4:
            raise Exception(f"No enough markers found. Required: 4,  found: {len(corners)}")

        ids = ids.flatten()
        refPts = []
        for i in (7,5,4,6):
            j = np.squeeze(np.where(ids == i))
            corner = np.squeeze(corners[j])
            refPts.append(corner)
        print(f"refPts: {refPts}")

        (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
        dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
        dstMat = np.array(dstMat)
        (srcH, srcW) = self.source.shape[:2]
        srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
        (H, _) = cv2.findHomography(srcMat, dstMat)
        warped = cv2.warpPerspective(self.source, H, (imgW, imgH))


        mask = np.zeros((imgH, imgW), dtype="uint8")
        cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255),
            cv2.LINE_AA)
        rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, rect, iterations=2)
        maskScaled = mask.copy() / 255.0
        maskScaled = np.dstack([maskScaled] * 3)
        warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
        imageMultiplied = cv2.multiply(image.astype(float), 1.0 - maskScaled)
        output = cv2.add(warpedMultiplied, imageMultiplied)
        output = output.astype("uint8")
        return output
