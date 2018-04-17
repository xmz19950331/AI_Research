"""
test the classification of given vgg features at given threshold range
"""

import pickle
import psycopg2
import os
import sys
import numpy as np
import cv2

abspath_2 = os.path.abspath(__file__)
dname_2 = os.path.dirname(abspath_2)
sys.path.append(dname_2+'/helpers/')

from SupportingClasses import BoundingBox, Frame
from tracking_func import tracking, test_tracking_thresholds
from runner import run

sys.path.append(dname_2+'/../../../triplet_loss/core')

from deep_matcher_wenxi import *




def classification_test_thresholds(bbox_triplets_list_file, low, high, step):
	''' Test the classification of given bounding box triplets at given threshold range. '''
	
	# set up the database connection
	try:
		conn = psycopg2.connect("dbname='larsde_images' user='flask' host='larsde.cs.columbia.edu' password='dvmm32123'")
		conn.autocommit= True
	except Exception, e:
		print e, "Connection Unsucessful"
	cur_images = conn.cursor()
	
	# initiate the learning model
	runner = run()	
	
	bbox_pairs_list = np.load(bbox_triplets_list_file)
	print len(bbox_pairs_list), "triplets"

	threshold_frame_pair = {}		
	for threshold in np.arange(low, high, step):
		print "test threshold: ", threshold
		result = test_bboxes_pairs(bbox_pairs_list, threshold, runner)
		
		
		
	#pickle.dump(data, open(os.path.join(data_dir, "results_0.8-1.3" + ".pickle"), "wb"))	
		
		
	
def test_bboxes_pairs(bbox_triplets, threshold, runner):
	''' Test the classification of bounding box triplets at given threshold. '''
	
	labels = []
	pred = []
	for tri in bbox_triplets:
		# test matching the target to the positive sample and negative sample
		pos_pred = runner.model.match_bbox_pairs(tri[0], tri[1], threshold)
		pred.append(pos_pred)
		labels.append(1)
		
		neg_pred = runner.model.match_bbox_pairs(tri[0], tri[2], threshold)
		pred.append(neg_pred)
		labels.append(0)
		
		sys.stdout.write(".")
		
	#print pred
	#print labels
	sys.stdout.write("\n")
	
	pred = np.asarray(pred)
	labels = np.asarray(labels)
	error_rate = 1 - sum(pred == labels) / float(len(pred))
	print "threshold", threshold, "has error rate", error_rate
	
	
if __name__ == "__main__":
	if len(sys.argv) != 5:
		print "This script tests the classification of given vgg features at given threshold range."
		print "usage: python vgg_model_test.py <triplets_match_list_path.npy> <threshold_low> <threshold_high> <threshold_step>"
		sys.exit()
	classification_test_thresholds(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))