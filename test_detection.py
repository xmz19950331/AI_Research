""" Test the detection of current caffe implementation as well as tensorflow implementation of faster rcnn. 
  1. Generate result text file
  2. use the reulst text file to compare with the ground truth annotations
"""

"""
Corrupted images in the dataset:
  495 2015-12-07 08:00:08
  263 2015-12-07 20:01:23
  734 2016-01-07 20:08:38
"""

import os, sys

# abspath_2 = os.path.abspath(__file__)
# dname_2 = os.path.dirname(abspath_2)
# sys.path.append(dname_2+'/helpers/')

# from extract_detection_tracking import detect_bounding_boxes
# from SupportingClasses import BoundingBox
# from timer import Timer

import pickle
import base64
import psycopg2
import cv2
from tqdm import tqdm
import numpy as np

IMAGE_FOLDER = '/home/mingzhi/detection_test_images'

# def generate_test_results():
    """Generate test results into a text file as format "time, camid, x1, y1, x2, y2\n" for each line
    """

#   # perform detection on the images
#   imgfiles = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if (os.path.isfile(os.path.join(IMAGE_FOLDER, f)) and f[-3:]=='jpg')]
#   num_images = len(imgfiles)

#   writef = open('text_files/caffe_frcnn_detection_results.txt', 'w')
#   bbox_list = []
#   T = Timer()
#   T.tic()
#   for i in tqdm(range(num_images)):
#     imgfile_name = imgfiles[i]
#     im = cv2.imread(imgfile_name)
#     extra_info = []
#     info_L = imgfile_name.split('_')
#     img_time = info_L[-1]
#     img_time = img_time[:-4]
#     img_time = img_time.replace('+', ':')
#     info_L = info_L[-2].split('/')
#     camid = info_L[-1]
#     bboxes = detect_bounding_boxes(im, [img_time, int(camid)])
#     for box in bboxes:
#       writef.write('{}, {:d}, {:d}, {:d}, {:d}, {:d}\n'.format(box.time, box.camid, box.x1, box.y1, box.x2, box.y2))
#     bbox_list.extend(bboxes)
#   T.toc()
#   print 'It takes {:.3f}s to detect {:d} images'.format(T.diff, num_images)
#   writef.close()


def compare_det_with_gt(dets, cur, thresh=0.5):
  """Compare detection results with the ground truth in this frame.
  """

  cur.execute("select coord_x1, coord_y1, coord_x2, coord_y2 from obj_annot where imgtime=%s and imgcamid=%s;", [dets[-1][0], dets[-1][1]])
  rows = cur.fetchall()

  TP = 0
  FP = 0
  FN = 0
  corrupted = 0

  for det in dets:
    for i in range(len(rows)):
      det_bbox = [int(v) for v in det[2:]]
      # print dets[-1][0], dets[-1][1]
      # print len(rows)
      # print i
      # print rows[i]
      if rows[i][0] is None or rows[i][1] is None or rows[i][2] is None or rows[i][3] is None:
        # print "corrupted annotation for {} at camid {}".format(dets[-1][0], dets[-1][1])
        corrupted += 1
        del rows[i]
        break
      gt_bbox = [int(v) for v in rows[i]]
      iou = cal_iou(det_bbox, gt_bbox)
      if iou >= thresh:
        TP += 1
        del rows[i]
        break
  
  FP = len(dets) - TP
  FN = len(rows)

  return TP, FP, FN, corrupted


def look_frame(camid, frametime, cur):
  """Exam and report the number of valid bbox in the frame.
  """

  cur.execute("select coord_x1, coord_y1, coord_x2, coord_y2 from obj_annot where imgtime=%s and imgcamid=%s;", [frametime, camid])
  rows = cur.fetchall()

  valid_bbox = 0
  for i in range(len(rows)):
    if rows[i][0] is None or rows[i][1] is None or rows[i][2] is None or rows[i][3] is None:
      continue
    valid_bbox += 1
  
  return valid_bbox, len(rows)

def cal_iou(det_bbox, gt_bbox):
  """Calculate IOU from two bounding boxes
  """

  det_ul = np.array(det_bbox[:2])
  det_br = np.array(det_bbox[2:])
  gt_ul = np.array(gt_bbox[:2])
  gt_br = np.array(gt_bbox[2:])

  _ul = np.maximum(det_ul, gt_ul)
  _br = np.minimum(det_br, gt_br)

  dist = _br - _ul
  dist = np.maximum(dist, 0)
  intersection = dist[0] * dist[1]

  det_dist = det_br - det_ul
  det_area = det_dist[0] * det_dist[1]

  gt_dist = gt_br - gt_ul
  gt_area = gt_dist[0] * gt_dist[1]

  return float(intersection) / float(det_area + gt_area - intersection)


def get_accuracy(resultfile, thresh=0.5):
  """Get the accuracy by comparing detection annotation file with the ground truth.
    Assume detection result file is ordered.
    The algo is robust as it address corrupted annotations and undetected frames.
  """

  # connect to the dbs
  try:
    conn = psycopg2.connect("dbname='larsde_other' user='flask' host='larsde.cs.columbia.edu' password='dvmm32123'")
    conn.autocommit= True
  except Exception, e:
    print e, "Connection Unsucessful"
  cur = conn.cursor()

  with open(resultfile, 'r') as rfile:
    results = rfile.readlines()

  # get all frames
  cur.execute("select distinct imgcamid, imgtime from obj_annot where objecttype='cars';")
  rows = cur.fetchall()
  all_frame_strs = []
  for row in rows:
    frame_str = str(row[0]) + "+" + str(row[1])
    all_frame_strs.append(frame_str)

  frame_num = 0
  gt_bbox_num = 0
  TP = 0
  FP = 0
  FN = 0
  corrupted = 0

  line_num = 0
  line = results[line_num]
  line = line[:-1]
  line = line.split(',')
  line = [i.strip() for i in line]
  cur_frame = [line]
  line_num += 1
  sys.stdout.write(".")
  sys.stdout.flush()
  while line_num < len(results):
    line = results[line_num]
    line = line[:-1]
    line = line.split(',')
    line = [i.strip() for i in line]
    if line[0] == cur_frame[-1][0] and line[1] == cur_frame[-1][1]:
      cur_frame.append(line)
    else:
      # compute and accumulate the error rate for the current frame
      all_frame_strs.remove(line[1]+"+"+line[0])
      cTP, cFP, cFN, cCorrupted = compare_det_with_gt(cur_frame, cur, thresh)
      TP += cTP
      FP += cFP
      FN += cFN
      corrupted += cCorrupted
      gt_bbox_num += cTP + cFN + cCorrupted
      cur_frame = [line]
      frame_num += 1

    line_num += 1
    sys.stdout.write(".")
    if (line_num%100)==0:
      sys.stdout.write(str(line_num))
    sys.stdout.flush()	
  
  # compute and accumulate the error rate for the current frame
  cTP, cFP, cFN, cCorrupted = compare_det_with_gt(cur_frame, cur, thresh)
  TP += cTP
  FP += cFP
  FN += cFN
  corrupted += cCorrupted
  gt_bbox_num += cTP + cFN + cCorrupted

  sys.stdout.write(".\n")
  sys.stdout.flush()

  # print all_frame_strs
  all_frame_strs.remove("495+2015-12-07 08:00:08")
  all_frame_strs.remove("263+2015-12-07 20:01:23")
  all_frame_strs.remove("734+2016-01-07 20:08:38")

  # get the missed bboxes
  total_missed_bbox = 0
  total_missed_valid_bbox = 0
  for remain_frame in all_frame_strs:
    frame = remain_frame.split('+')
    valid_missed_bbox, missed_bbox = look_frame(frame[0], frame[1], cur)
    total_missed_valid_bbox += valid_missed_bbox
    total_missed_bbox += missed_bbox

  print "It missed {:d} frames which has {:d} valid bounding boxes".format(len(all_frame_strs), total_missed_valid_bbox)

  print "In {:d} frames, there are {:d} gt bounding boxes with {:d} corrupted annotations and TP={:d}, FP={:d}, FN={:d}".format(frame_num, gt_bbox_num+total_missed_bbox, corrupted, TP, FP, FN+total_missed_valid_bbox)

if __name__ == "__main__":
  # generate_test_results()
  # resultfile = '../detection_test_results/tensorflow_frcnn_detection_results.txt'
  resultfile = '../detection_test_results/caffe_frcnn_detection_results.txt'
  get_accuracy(resultfile)

   