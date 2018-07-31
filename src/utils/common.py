import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')



BRISK_THRESHOLD = 40
BRISK_MIN_VALUE = 10**-3

KALMAN_LIKELIHOOD_THRESHOLD = 10**7
KALMAN_SURE_THRESHOLD = 10**6

#If Brisk finds no matches but Kalman is very sure then accept match
INDEX_SELECTOR_THRESHOLD = 0#5000#(KALMAN_LIKELIHOOD_THRESHOLD-KALMAN_SURE_THRESHOLD) *BRISK_MIN_VALUE  
