

def camshift(self):
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2, 1 )
    print("performing camshift")
    count = 1
    while(1):#self.stop == False):
        count += 1
        start = time.time()
        if(len(self.buffers["camshift"]) > 0):
            #UPDATE CAMSHIFT BUFFER WITH NEW VALUES FROM SEG THREAD
            self.buffer_locks["camshift"].acquire()
            if(self.camshift_update):
                self.track("camshift")
                self.camshift_update = False

            image = self.buffers["image"][0]

            for roi_index in range(len(self.buffers["camshift"][0].roi)):

                new_roi = self.buffers["camshift"][0]
                old_roi = new_roi.roi[roi_index]
                hist = self.buffers["camshift"][0].hist[roi_index]

                #print(hist)
                ''''ry1, rx1, ry2, rx2'''
                x1,x2 = min(old_roi[1],old_roi[3]),max(old_roi[1],old_roi[3])
                y1,y2 = min(old_roi[2],old_roi[0]),max(old_roi[2],old_roi[0])

                #get coords of top edge,height, left edge, width
                r,h = y1,y2-y1
                c,w = x1,x2-x1

                #reduce box size to focus on peron/object trunk
                div = 3
                r,h,c,w = int(r+h/2-h/div),int(h/div),int(c+w/2-w/div),int(w/div)

                track_window = (c,r,w+1,h+1)

                print("points are : [()" + str(old_roi[1]) + ',' + str(old_roi[0]) + '),(' + str(old_roi[3]) + ',' + str(old_roi[2]) + ")]")
                print("initial track_window is : " + str(track_window))
                print("image size is " + str(image.frame.shape))

                roi = image.frame[r:r+h, c:c+w]
                hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                if all([ v > 0 for v in roi.shape ]):
                    print("roi shape is : " + str(roi.shape))
                    mask = cv2.inRange(hsv_roi, np.array([0., 60.,32.]), np.array([180.,255.,255.]))

                    if(type(hist) != np.ndarray):
                        hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
                        new_roi.hist[roi_index] = hist

                    cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)

                    hsv = cv2.cvtColor(image.frame, cv2.COLOR_BGR2HSV)
                    dst = cv2.calcBackProject([hsv],[0],hist,[0,180],1)
                    # apply meanshift to get the new location
                    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
                    print("New track window is "  + str(track_window))
                    [c,r,w,h] = track_window

                    #re-calculate box
                    r,h,c,w = int(r+(h)-(h*div/2)),h*div,int(c+w-(w*div/2)),w*div
                    new_roi.roi[roi_index] = [c,c+w,r,r+h]
                    self.buffers["camshift"].append(new_roi)
                    #self.track("camshift")

            self.buffer_locks["camshift"].release()
            end = time.time()
            print("Camshift took around : " + str(end-start))
            time.sleep(0.1)
        else:
            print("Buffer is not long enough")
            time.sleep(1)
