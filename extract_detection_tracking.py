import psycopg2
import os
import sys
import io
import cv2
import pickle
import argparse
import numpy as np
from multiprocessing import Pool

from tqdm import tqdm

abspath_2 = os.path.abspath(__file__)
dname_2 = os.path.dirname(abspath_2)
sys.path.append(dname_2+'/helpers/')

from SupportingClasses import BoundingBox, Frame
# from utils.timer import Timer
from detection_func import im_detect, net, CLASSES, nms
from tracking_func import tracking
from runner import run
from timer import Timer

# log_file_name = "data/database.log"
# if os.path.isfile(log_file_name): 
# 	with open("data/database.log", 'r') as f:
# 		content = f.read()
# 	LabelID = long(content.split()[0].strip()) + 1
# else:
LabelID = long(1)

runner = run()
corrupted_imgs = []


def detect_bounding_boxes(im, extra_info=None):
	''' detect bounding boxes of cars in the given image. 
	
	Args:
		im: frame image from opencv imread.
		extra_info: List of information about the frame images, in format [str(datetime), int(camid)]
	Returns:
		list of BoundingBox.
	'''
	
	global LabelID
	
	# use detection function from faster-rcnn model
	T = Timer()
	T.tic()
	scores, boxes = im_detect(net, im)
	T.toc()
	print 'detection takes {:.3f}s'.format(T.diff)
	
	CONF_THRESH = 0.9
	NMS_THRESH = 0.3
	
	bbox_list = []
	
	for cls_ind, cls in enumerate(CLASSES[1:]):
		cls_ind += 1 # because we skipped background
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
		keep = nms(dets, NMS_THRESH)
		dets = dets[keep, :]
		# vis_detections(im, cls, dets, image_dir, image_name, thresh=CONF_THRESH)
		# print dets
		inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
		for i in inds:
			bbox_coords = dets[i, :4]
			# print bbox_coords
			bbox = BoundingBox(LabelID, int(bbox_coords[0]), int(bbox_coords[1]), int(bbox_coords[2]), int(bbox_coords[3]))
			if extra_info is not None:
				bbox.time = extra_info[0]
				bbox.camid = extra_info[1]
			bbox_list.append(bbox)
			LabelID += 1
		# return dets[inds, :4]
	return bbox_list
	
	

def save_bboxes_to_db(cur, bbox_list, time, camid, table_name):
	''' Save a list of bounding boxes to database '''
	
	count = 0
	for bbox in bbox_list:
		try:                
			# cur.execute("INSERT INTO " + table_name + 
			# 	" (labelid, imgtime, imgcamid, coord_x1, coord_y1, coord_x2, coord_y2, objecttype) VALUES (%s, %s, %s, %s, %s, %s, %s, 'car');", 
			# 	[bbox.labelid, time, camid, bbox.x1, bbox.y1, bbox.x2, bbox.y2])
			cur.execute("INSERT INTO " + table_name + 
				" (imgtime, imgcamid, coord_x1, coord_y1, coord_x2, coord_y2, objecttype) VALUES (%s, %s, %s, %s, %s, %s, 'car');", 
				[time, camid, bbox.x1, bbox.y1, bbox.x2, bbox.y2])
			count += 1
		except Exception, e:
			print ("Error writing to database: " + str(e))	
	
	# print "saved", count, "bounding boxes."



def save_tracking_to_db(cur, pair_list, table_name):
	''' Save a list of tracked bounding box pairs to database '''
	
	count = 0
	for pair in pair_list:
		try:                
			cur.execute("INSERT INTO " + table_name + " (labelid1, labelid2) VALUES (%s, %s);", [pair[0], pair[1]])
			count += 1
		except Exception, e:
			print ("Error writing to database: " + str(e))	
	
	print "saved", count, "tracked pairs."	


def extract_detection_only(cur, cur_images, camid, starttime, endtime, image_dir, limit, tablename):
	''' Only extract the detected bounding boxes and populate it to the database.

	Args:
		cur: db cursor of larsde_other database
		cur_images: db cursor of larsde_images database
		camid: camera id
		starttime: unicode start time for detection extraction in 
			datatime format e.g., unicode('12/07/2015, 08:00:00 AM')
		endtime: unicode end time for detection extraction
		image_dir: temporary image directory
		limit: number of image to extract at most
	'''

	global runner
	global corrupted_imgs
		
	cur_images.execute("SELECT content, time from images WHERE new_id=%s AND time > %s AND time <= %s order by time asc LIMIT %s;", 
					   [camid, starttime, endtime, 10])    

	data = cur_images.fetchall()
	
	L = []
	count = 0
	for row_ind in range(len(data)):
		# print "this is row", row_ind
		row = data[row_ind]
		fh = open(os.path.join(image_dir, "temp.jpg"),'wb')
		fh.write(row[0])
		fh.close()	
		im = cv2.imread(os.path.join(image_dir, "temp.jpg"))
		
		if (im is None or len(im) == 0):
			corrupted_imgs.append(str(camid) + "@" + str(row[1]))
			continue
		
		# detect bounding boxes
		bbox_list = detect_bounding_boxes(im)

		# save the bounding boxes of the current frame to database
		save_bboxes_to_db(cur, bbox_list, row[1], camid, tablename)
		count += 1
		if count == limit:
			break



def extract_detection_and_tracking(cur, cur_images, camid, starttime, endtime, image_dir):
	''' Extract detected bounding boxes and tracking from the given period of time and camid.
		Save both information to the database.

	Args:
		cur: db cursor of larsde_other database
		cur_images: db cursor of larsde_images database
		camid: camera id
		starttime: unicode start time for detection extraction in 
			datatime format e.g., unicode('12/07/2015, 08:00:00 AM')
		endtime: unicode end time for detection extraction
		image_dir: temporary image directory 
	'''
	global runner
	global corrupted_imgs
		
	cur_images.execute("SELECT content, time from images WHERE new_id=%s AND time > %s AND time <= %s order by time asc;", [camid, starttime, endtime])    

	data = cur_images.fetchall()
	
	# count = 0;
	last_frame_time = None;
	last_bbox_list = None;
	for row_ind in range(len(data)):
		print "this is row", row_ind
		row = data[row_ind]
		fh = open(os.path.join(image_dir, "temp.jpg"),'wb')
		fh.write(row[0])
		fh.close()	
		im = cv2.imread(os.path.join(image_dir, "temp.jpg"))
		
		if (im is None or len(im) == 0):
			corrupted_imgs.append(str(camid) + "@" + str(row[1]))
			continue
		
		# detect bounding boxes
		bbox_list = detect_bounding_boxes(im)

		# save the bounding boxes of the current frame to database
		save_bboxes_to_db(cur, bbox_list, row[1], camid, "extracted_bounding_boxes")
		
		if row_ind != 0:
			# get tracking results from the current frame and the previous frame
			frame1 = Frame(camid, last_frame_time)
			frame1.bboxes = last_bbox_list[:]
			frame2 = Frame(camid, row[1])
			frame2.bboxes = bbox_list
			
			# get the object vector map
			try:
				cur.execute("SELECT vecmap from obj_velocity_0730_25min WHERE camid=%s LIMIT 1;", [camid])
			except Exception, e:
				print e, "Connection Unsucessful"
			vecmap_row = cur.fetchone()
			buf = io.BytesIO()
			buf.write(vecmap_row[0])
			buf.seek(0)
			im_velocity_array = np.load(buf)			
			# use a threshold of 1.25
			found_pairs, empty, wrong_bboxes, pair_used_bboxes = tracking(runner, frame1, frame2, cur_images, 1.25, velocity_array=im_velocity_array)		
			save_tracking_to_db(cur,found_pairs, "extracted_tracking")
			
		last_frame_time = row[1]
		last_bbox_list = bbox_list[:]
		
		# count += 1
	

def populate_bbox_for_heatmap(cur, cur_images, tablename, camid_start, camid_end):
	"""Extract bboxs from detection and then populate them to database for heatmap visualization.

	Args:
		cur: db cursor of larsde_other database
		cur_images: db cursor of larsde_images database
	"""

	with open('pickles/camera_list.pickle', 'rb') as handle:
		cameras = pickle.load(handle)
	
	# starttime = unicode('12/07/2015, 08:00:00 AM')
	# endtime = unicode('12/07/2015, 08:29:59 AM')	
	image_dir = "temp_images"	
	
	# extract detection and tracking for all cameras
	for i in tqdm(xrange(camid_start, camid_end+1), desc="cameras"):
		camid = cameras[i]
		# for m in xrange(2):
		for d in tqdm(xrange(1, 32), desc="days"):
			for clock in xrange(2):
				for h in xrange(12):
					# if m == 0:
					# 	mm = '12'
					# 	yy = '2015'
					# else:
					# 	mm = '01'
					# 	yy = '2016'
					if clock == 0:
						cc = 'AM'
					else:
						cc = 'PM'

					starttime = unicode('3/'+str(d)+'/2016, '+str(h)+':00:00 '+cc)
					endtime = unicode('3/'+str(d)+'/2016, '+str(h)+':59:59 '+cc)

					# extract_detection_and_tracking(cur, cur_images, camid, starttime, endtime, image_dir)
					extract_detection_only(cur, cur_images, camid, starttime, endtime, image_dir, 1, tablename)
		
		# with open("data/database.log", 'w') as f:
		# 	f.write(str(LabelID)+" "+str(i))
		# print "==============finish", i+1, "cameras==============="
		
	print corrupted_imgs
		
def multiprocess_db_population(cur, cur_images):
	"""Populate the database using multi-processing.
	"""

	pool = Pool()
	result1 = pool.apply_async(populate_bbox_for_heatmap, [cur, cur_images, "bbox_week_cam_0_19", 0, 19])
	result2 = pool.apply_async(populate_bbox_for_heatmap, [cur, cur_images, "bbox_week_cam_20_39", 20, 39])
	result3 = pool.apply_async(populate_bbox_for_heatmap, [cur, cur_images, "bbox_week_cam_40_59", 40, 59])
	result4 = pool.apply_async(populate_bbox_for_heatmap, [cur, cur_images, "bbox_week_cam_60_79", 60, 79])
	result5 = pool.apply_async(populate_bbox_for_heatmap, [cur, cur_images, "bbox_week_cam_80_99", 80, 99])
	result6 = pool.apply_async(populate_bbox_for_heatmap, [cur, cur_images, "bbox_week_cam_100_119", 100, 119])
	result7 = pool.apply_async(populate_bbox_for_heatmap, [cur, cur_images, "bbox_week_cam_120_139", 120, 139])
	result8 = pool.apply_async(populate_bbox_for_heatmap, [cur, cur_images, "bbox_week_cam_140_159", 140, 159])
	result9 = pool.apply_async(populate_bbox_for_heatmap, [cur, cur_images, "bbox_week_cam_160_187", 160, 187])

	print result1.get()
	print result2.get()
	print result3.get()
	print result4.get()
	print result5.get()
	print result6.get()
	print result7.get()
	print result8.get()
	print result9.get()
	print result1.successful()
	print result2.successful()
	print result3.successful()
	print result4.successful()
	print result5.successful()
	print result6.successful()
	print result7.successful()
	print result8.successful()
	print result9.successful()


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='extract bounding boxes or tracking results')

	parser.add_argument('to_extract', help='decide which to extract, input can be "bbox" or "tracking"')
	parser.add_argument('tablename', help='name of database table the extracted will be save to')
	parser.add_argument('camid_start', type=int, help='start index of camid to extract')
	parser.add_argument('camid_end', type=int, help='end index of camid to extract')
	args = parser.parse_args()

	# set up the database connection
	try:
		conn_images = psycopg2.connect("dbname='larsde_images' user='flask' host='larsde.cs.columbia.edu' password='dvmm32123'")
		conn_images.autocommit= True
		conn = psycopg2.connect("dbname='larsde_other' user='flask' host='larsde.cs.columbia.edu' password='dvmm32123'")
		conn.autocommit= True
	except Exception, e:
		print e, "Connection Unsucessful"
	cur_images = conn_images.cursor()
	cur = conn.cursor()
	
	if args.to_extract == 'bbox':   
		populate_bbox_for_heatmap(cur, cur_images, args.tablename, args.camid_start, args.camid_end)

	elif args.to_extract == 'multi':
		multiprocess_db_population(cur, cur_images)
	
	