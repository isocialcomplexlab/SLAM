

import rospy

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from nav_msgs.msg import Odometry

from yolov5.detect import inference

import pandas as pd
import numpy as np

import cv2

import time

MODEL = './yolov5/runs/train/model/weights/best.pt'

# Calculates Distance
def get_dist(x1, x2, y1, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

# Check Overlapping Bounding Boxes
def rect_overlap(rec1_x1, rec1_x2, rec1_y1, rec1_y2, 
                 rec2_x1, rec2_x2, rec2_y1, rec2_y2):
    if (rec2_x2 > rec1_x1 and rec2_x2 < rec1_x2) or (rec2_x1 > rec1_x1 and rec2_x1 < rec1_x2):
        x_ovr = True
    else:
        x_ovr = False

    if (rec2_y2 > rec1_y1 and rec2_y2 < rec1_y2) or (rec2_y1 > rec1_y1 and rec2_y1 < rec1_y2):
        y_ovr = True
    else:
        y_ovr = False
    if x_ovr and y_ovr:
        return True
    else:
        return False

# Check Intersections for Lab Entry
def verify_entrance(objs, boxes, body, face):
    body_x1 = body[0]
    body_y1 = body[1]
    body_x2 = body[2]
    body_y2 = body[3]

    face_x1 = face[0]
    face_y1 = face[1]
    face_x2 = face[2]
    face_y2 = face[3]

    allow_entrance = True

    ## Backpack Intersection
    if 0 in objs['obrig'] or 0 in objs['proib']:
        backpacks_overlap = []
        for backpack in list(filter(lambda x: x[5] == 0, boxes)):
            backpack_x1 = backpack[0]
            backpack_y1 = backpack[1]
            backpack_x2 = backpack[2]
            backpack_y2 = backpack[3]

            backpack_overlap = rect_overlap(body_x1, body_x2, body_y1, body_y2, 
                                            backpack_x1, backpack_x2, backpack_y1, backpack_y2)
            
            backpacks_overlap.append(backpack_overlap)

        has_backpack = any(backpacks_overlap)
            
        if 0 in objs['obrig']:
            if has_backpack:
                print('MOCHILA OBRIGATORIA DETECTADA')
            else:
                print('MOCHILA OBRIGATORIA NAO DETECTADA')
                allow_entrance = False
        
        elif 0 in objs['proib']:
            if not has_backpack:
                print('MOCHILA PROIBIDA NAO DETECTADA')
            else:
                print('MOCHILA PROIBIDA DETECTADA')
                allow_entrance = False

    ## Bottle Intersection
    if 1 in objs['obrig'] or 1 in objs['proib']:
        bottles_overlap = []
        for bottle in list(filter(lambda x: x[5] == 1, boxes)):
            bottle_x1 = bottle[0]
            bottle_y1 = bottle[1]
            bottle_x2 = bottle[2]
            bottle_y2 = bottle[3]

            bottle_overlap = rect_overlap(body_x1, body_x2, body_y1, body_y2, 
                                          bottle_x1, bottle_x2, bottle_y1, bottle_y2)
            
            bottles_overlap.append(bottle_overlap)

        has_bottle = any(bottles_overlap)
            
        if 1 in objs['obrig']:
            if has_bottle:
                print('GARRAFA OBRIGATORIA DETECTADA')
            else:
                print('GARRAFA OBRIGATORIA NAO DETECTADA')
                allow_entrance = False
        
        elif 1 in objs['proib']:
            if not has_bottle:
                print('GARRAFA PROIBIDA NAO DETECTADA')
            else:
                print('GARRAFA PROIBIDA DETECTADA')
                allow_entrance = False
            
    ## Protective Goggles Intersection
    if 3 in objs['obrig'] or 3 in objs['proib']:
        eyewears_overlap = []
        for eyewear in list(filter(lambda x: x[5] == 3, boxes)):
            eyewear_x1 = eyewear[0]
            eyewear_y1 = eyewear[1]
            eyewear_x2 = eyewear[2]
            eyewear_y2 = eyewear[3]

            eyewear_overlap = rect_overlap(face_x1, face_x2, face_y1, face_y2, 
                                           eyewear_x1, eyewear_x2, eyewear_y1, eyewear_y2)
            
            eyewears_overlap.append(eyewear_overlap)

        has_eyewear = any(eyewears_overlap)
            
        if 3 in objs['obrig']:
            if has_eyewear:
                print('OCULOS DE PROTECAO OBRIGATORIO DETECTADO')
            else:
                print('OCULOS DE PROTECAO OBRIGATORIO NAO DETECTADO')
                allow_entrance = False
        
        elif 3 in objs['proib']:
            if not has_eyewear:
                print('OCULOS DE PROTECAO PROIBIDO NAO DETECTADO')
            else:
                print('OCULOS DE PROTECAO PROIBIDO DETECTADO')
                allow_entrance = False

    ## Fruit Intersection
    if 5 in objs['obrig'] or 5 in objs['proib']:
        fruits_overlap = []
        for fruit in list(filter(lambda x: x[5] == 5, boxes)):
            fruit_x1 = fruit[0]
            fruit_y1 = fruit[1]
            fruit_x2 = fruit[2]
            fruit_y2 = fruit[3]

            fruit_overlap = rect_overlap(body_x1, body_x2, body_y1, body_y2, 
                                         fruit_x1, fruit_x2, fruit_y1, fruit_y2)
            
            fruits_overlap.append(fruit_overlap)

        has_fruit = any(fruits_overlap)
            
        if 5 in objs['obrig']:
            if has_fruit:
                print('FRUTA OBRIGATORIA DETECTADA')
            else:
                print('FRUTA OBRIGATORIA NAO DETECTADA')
                allow_entrance = False
        
        elif 5 in objs['proib']:
            if not has_fruit:
                print('FRUTA PROIBIDA NAO DETECTADA')
            else:
                print('FRUTA PROIBIDA DETECTADA')
                allow_entrance = False

    ## Jaleco Intersection
    if 6 in objs['obrig'] or 6 in objs['proib']:
        labcoats_overlap = []
        for labcoat in list(filter(lambda x: x[5] == 6, boxes)):
            labcoat_x1 = labcoat[0]
            labcoat_y1 = labcoat[1]
            labcoat_x2 = labcoat[2]
            labcoat_y2 = labcoat[3]

            labcoat_overlap = rect_overlap(body_x1, body_x2, body_y1, body_y2, 
                                           labcoat_x1, labcoat_x2, labcoat_y1, labcoat_y2)
            
            labcoats_overlap.append(labcoat_overlap)

        has_labcoat = any(labcoats_overlap)
            
        if 6 in objs['obrig']:
            if has_labcoat:
                print('JALECO OBRIGATORIO DETECTADO')
            else:
                print('JALECO OBRIGATORIO NAO DETECTADO')
                allow_entrance = False
        
        elif 6 in objs['proib']:
            if not has_labcoat:
                print('JALECO PROIBIDO NAO DETECTADO')
            else:
                print('JALECO PROIBIDO DETECTADO')
                allow_entrance = False

    ## Mask Intersection
    if 7 in objs['obrig'] or 7 in objs['proib']:
        masks_overlap = []
        for mask in list(filter(lambda x: x[5] == 7, boxes)):
            mask_x1 = mask[0]
            mask_y1 = mask[1]
            mask_x2 = mask[2]
            mask_y2 = mask[3]

            mask_overlap = rect_overlap(face_x1, face_x2, face_y1, face_y2, 
                                        mask_x1, mask_x2, mask_y1, mask_y2)
            
            masks_overlap.append(mask_overlap)

        has_mask = any(masks_overlap)
            
        if 7 in objs['obrig']:
            if has_mask:
                print('MASCARA OBRIGATORIA DETECTADA')
            else:
                print('MASCARA OBRIGATORIA NAO DETECTADA')
                allow_entrance = False
        
        elif 7 in objs['proib']:
            if not has_mask:
                print('MASCARA PROIBIDA NAO DETECTADA')
            else:
                print('MASCARA PROIBIDA DETECTADA')
                allow_entrance = False

    return allow_entrance

# Functions for Creating the Semantic Map
class SemanticInfo(object):
    def __init__(self):
        self.df = pd.read_csv('labs.csv', sep=';')
        self.df['itens_obrig'] = self.df['itens_obrig'].astype(str)
        self.df['itens_proib'] = self.df['itens_proib'].astype(str)
        
        
        self.marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        self.classes = ['Backpack', 'Bottle', 'Door', 'Eyewear', 'Face', 'Fruit', 'LabCoat',
                        'Mask', 'Person']
        self.doors = []

        self.pose = {}

        self.markerArray = MarkerArray()

        self.ditance = []

    # Search Items in the Database
    def get_items(self, lab):
        print(self.df.loc[self.df['lab_num'] == lab]['itens_obrig'].str.split(','))
        obr_items = [int(float(x)) for x in self.df.loc[self.df['lab_num'] == lab]['itens_obrig'].str.split(',').iloc[0]]
        proi_items = [int(float(x)) for x in self.df.loc[self.df['lab_num'] == lab]['itens_proib'].str.split(',').iloc[0]]

        return {
            'obrig' : obr_items,
            'proib' : proi_items
        }
        
    # Labels on the Map Laboratories already Identified
    def publish_init_doors(self):
        for door in self.df[~(pd.isna(self.df['p_x']))].to_dict('records'):
            self.publish_door(door['lab_num'], door['p_x'], door['p_y'])
    
    # Add Identified Laboratory Information
    def add_door(self, lab, x, y):
        if (self.pose['x'] < 0):
            x = -x
        self.df.loc[self.df['lab_num'] == lab, 'p_x'] = self.pose['x'] + x
        self.df.loc[self.df['lab_num'] == lab, 'p_y'] = self.pose['y'] + y
        self.df.loc[self.df['lab_num'] == lab, 't_x'] = self.pose['x']
        self.df.loc[self.df['lab_num'] == lab, 't_y'] = self.pose['y']

        self.df.to_csv('labs.csv', sep=';', index=False)
        
        self.publish_door(lab, self.pose['x'] + x, self.pose['y'] + y)
    
    # Label the Lab on the Map
    def publish_door(self, lab, x, y):
        mkr = self.get_marker(lab, x, y)
        self.markerArray.markers.append(mkr)
        
        self.publish_markers()
    
    # Check Distance Between Doors
    def check_close_door(self, x, y):
        min_dist = np.inf
        min_lab = -1
        for door in self.df[~(pd.isna(self.df['p_x']))].to_dict('records'): 
            dist = get_dist(door['p_x'], self.pose['x'] + x, door['p_y'], self.pose['y'] + y)
            print('DISTANCIA PORTA:')
            print(dist)
            if dist < 2.0 and dist < min_dist:
                min_dist = dist
                min_lab = door['lab_num']
        return min_lab
    
    # Check Distance from Robot to Door
    def check_door_distance_to_mark(self):
        for door in self.df[~(pd.isna(self.df['p_x']))].to_dict('records'): 
            dist = get_dist(door['t_x'], self.pose['x'], door['t_y'], self.pose['y'])
            print('DISTANCIA:')
            print(dist)
            if dist < 1.5:
                return False
        return True
    
    # Create Marker Object
    def get_marker(self, id, x, y):
        marker = Marker()

        marker.id = id

        marker.header.frame_id = "/map"
        marker.header.stamp = rospy.Time.now()

        marker.type = 1
        marker.action = Marker.ADD

        marker.scale.x = 0.3
        marker.scale.y = 0.9
        marker.scale.z = 2.0

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 1.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        return marker

    # Publish Markers on the Map
    def publish_markers(self):
        self.marker_pub.publish(self.markerArray)
        rospy.rostime.wallsleep(0.5)

    # Saves Distances Between Robot and Obstacles
    def distance_callback(self, msg):
        try:
            self.ditance = msg.ranges
        except Exception as e:
            print(e)

    # Save Robot Position
    def pose_callback(self, msg):
        try:
            self.pose['x'] = msg.pose.pose.position.x
            self.pose['y'] = msg.pose.pose.position.y
        except Exception as e:
            print(e)

    
if __name__ == '__main__':
    img_path = 'images/camera_image.jpeg'

    si = SemanticInfo()

    rospy.init_node('rviz_marker')

    init = False
    while not rospy.is_shutdown():
        if not init:
            rospy.rostime.wallsleep(0.5)
            si.publish_init_doors()
            init = True
            
        rospy.Subscriber('/scan', LaserScan, si.distance_callback)
        rospy.Subscriber('/odom', Odometry, si.pose_callback)

        dists = si.ditance

        try:
            boxes, im0 = inference(classes = [2], 
                                    source = img_path, 
                                    weights = MODEL, 
                                    conf_thres = 0.85, 
                                    iou_thres = 0.85)
        except:
            continue
        
        cv2.imshow(str(img_path), im0)
        cv2.waitKey(100)

        if len(boxes) > 0 and len(dists) > 0:
            for box in boxes[0]:
                x1 = box[0]
                y1 = box[1]

                x2 = box[2]
                y2 = box[3]

                perc = box[4]

                med = (x1 + x2) / 2
                
                if med <= 320:
                    med = -(320 - med)
                else:
                    med = med - 320
                 
                closest_lab = si.check_close_door(dists[320 + int(med)], med/640)
                closest_lab = 1
                if closest_lab != -1:
                    res = ''
                    while res not in ['s', 'n']:
                        print('Você está próximo ao laboratório '+ str(closest_lab) + ', começar a identificar pessoas? (S/N)')
                        res = input()

                    if res == 's':
                        while True:
                            objs = si.get_items(int(closest_lab))
                            classes = objs['obrig'] + objs['proib'] + [4, 8]

                            try:
                                people_boxes, im_people = inference(classes = classes,
                                                                    source = img_path,
                                                                    weights = MODEL,
                                                                    conf_thres = 0.25,
                                                                    iou_thres = 0.25)                   
                            except:
                                continue

                            cv2.imshow(str(img_path), im_people)
                            p_key = cv2.waitKey(33)
                            
                            if p_key == 27:
                                break

                            if len(people_boxes) > 0:
                                people_boxes = people_boxes[0]

                                bodies = list(filter(lambda x: x[5] == 8, people_boxes))
                                faces = list(filter(lambda x: x[5] == 4, people_boxes))

                                if len(bodies) > 0 and len(faces) > 0:
                                    body = bodies[0]
                                    face = faces[0]
                                    
                                    allow_entrance = verify_entrance(objs, people_boxes, body, face)

                                    if allow_entrance:
                                        print('ENTRADA AUTORIZADA')
                                    else:
                                        print('ENTRADA DESAUTORIZADA')

                                    time.sleep(5.0)

                    elif res == 'n':
                        break

                if si.check_door_distance_to_mark():
                    lab = -1
                    while int(lab) not in list(si.df['lab_num']):
                        print('Qual laboratório foi detectado?')
                        lab = input()
                        if lab =='n':
                            break

                    res = ''
                    if lab != 'n':
                        si.add_door(int(lab), dists[320 + int(med)], med/640)

                        while res not in ['s', 'n']:
                            print('Começar a identificar pessoas? (S/N)')
                            res = input()
                        
                        if res == 's' or res == 'S':
                            while True:
                                objs = si.get_items(int(lab))
                                classes = objs['obrig'] + objs['proib'] + [4, 8]
                                print(classes)

                                try:
                                    people_boxes, im_people = inference(classes = classes,
                                                                        source = img_path,
                                                                        weights = MODEL,
                                                                        conf_thres = 0.5,
                                                                        iou_thres = 0.5)                                
                                
                                except:
                                    continue

                                cv2.imshow(str(img_path), im_people)
                                p_key = cv2.waitKey(33)
                                
                                if p_key == 27:
                                    break

                                if len(people_boxes) > 0:
                                    people_boxes = people_boxes[0]

                                    bodies = list(filter(lambda x: x[5] == 8, people_boxes))
                                    faces = list(filter(lambda x: x[5] == 4, people_boxes))

                                    if len(bodies) > 0 and len(faces) > 0:
                                        body = bodies[0]
                                        face = faces[0]
                                        
                                        allow_entrance = verify_entrance(objs, people_boxes, body, face)

                                        if allow_entrance:
                                            print('ENTRADA AUTORIZADA')
                                        else:
                                            print('ENTRADA DESAUTORIZADA')

                                        time.sleep(5.0)

                    elif res == 'n' or res == 'N':
                        break
                    else:
                        continue

