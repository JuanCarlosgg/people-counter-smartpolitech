import os
import sys
import traceback
import multiprocessing
import time
import signal
import traceback

import cv2
import numpy as np
import requests

from deep_sort_pytorch.deep_sort import build_tracker
from deep_sort_pytorch.detector import build_detector
from deep_sort_pytorch.utils.draw import draw_boxes
from deep_sort_pytorch.utils.parser import get_config

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import argparse
from argparse import RawTextHelpFormatter


class points():
    def __init__(self, p11, p12, p21, p22):
        self.p11 = p11
        self.p12 = p12
        self.p21 = p21
        self.p22 = p22

class Detector(multiprocessing.Process):

    def __init__(self, task_queue, result_queue, peopleInside_q, reg):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.regiones = reg
        self.width = None
        self.peopleInside = peopleInside_q

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cfg = get_config()
        cfg.merge_from_file("deep_sort_pytorch/configs/yolov3.yaml")
        cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        detector = build_detector(cfg, use_cuda=True)
        deepsort = build_tracker(cfg, use_cuda=True)
        identities = {}

        while True:
            try:

                #frame = self.task_queue.get(block=True, timeout=0.5)
                frame = self.task_queue.get()
                #print(f"{self.name} frames en espera: {self.task_queue.qsize()}")
                if self.width is None:
                    print(frame.shape)
                    self.width, self.height = frame.shape[:-1]
                    p11 = (0, self.regiones[0])
                    p21 = (self.height, self.regiones[0])
                    p12 = (0, int(self.regiones[1]))
                    p22 = (self.height, int(self.regiones[1]))
                    puntos = points(p11, p12, p21, p22)
                    Pmedio = int((p11[1] + p12[1])/2)
                    medias = Pmedio

                appearances = []

                im = frame[:, :, (2, 1, 0)]
                bbox_xywh, cls_conf, cls_ids = detector(im)
                if bbox_xywh is not None:
                    mask = cls_ids == 0
                    bbox_xywh = bbox_xywh[mask]
                    bbox_xywh[:, 3] *= 1.1
                    cls_conf = cls_conf[mask]
                    outputs = deepsort.update(bbox_xywh, cls_conf, im)
                    if len(outputs) > 0:
                        for output in outputs:
                            bbox_xyxy = output[:4]
                            x1, y1, x2, y2 = tuple(bbox_xyxy)
                            M = cv2.moments(frame[y1:y2, x1:x2, 0])
                            cX = int(M["m10"] / M["m00"]) + x1
                            cY = int(M["m01"] / M["m00"]) + y1
                            cv2.circle(frame, (cX, cY), 5, (255, 0, 255), -1)
                            id = output[-1]
                            if identities.get(id):
                                dst = np.linalg.norm(
                                    np.array((cX, cY) - identities[id][1]))
                                dst = dst if cY < identities[id][1][1] else -dst
                                identities[id] = (
                                    identities[id][0] + dst, np.array((cX, cY)), identities[id][1])
                            else:
                                identities[id] = (
                                    0, np.array((cX, cY)), None)
                            if puntos.p11[1]*0.9 < identities[id][1][1] and puntos.p12[1]*1.1 > identities[id][1][1]:
                                appearances.append(id)
                        bbox_xyxy = outputs[:, :4]
                        ids = outputs[:, -1]
                        frame = draw_boxes(
                            frame, bbox_xyxy, ids, offset=(0, 0))


                identities = {app : identities[app] for app in appearances}

                cv2.line(frame, puntos.p11, puntos.p21, (0,255,0), 2)
                cv2.line(frame, puntos.p12, puntos.p22, (0,255,0), 2)
                cv2.line(frame, (0, medias), (self.height, medias), (0,0,255), 3)

                for item in identities:

                    #print(f" {self.name}     posiciones", medias,  identities[item][1], identities[item][2],  identities[item][0] )
                    #print(medias >= identities[item][1][1] , identities[item][0] > 0, identities[item][2][1] > medias if identities[item][2] is not None else False, self.regiones[2])
                    #print(medias <= identities[item][1][1] , identities[item][0] < 0,  identities[item][2][1] < medias if identities[item][2] is not None else False, self.regiones[2])
   
                    if medias >= identities[item][1][1] and identities[item][0] > 5 and identities[item][2] is not None and identities[item][2][1] > medias: # >5
                        status = "¡¡¡ Saliendo !!!" if self.regiones[2] == 1 else "¡¡¡ Entrando !!!"
                        print(f" {self.name}: {status}, {identities[item][0]}")
                        self.result_queue.put(-1*self.regiones[2])
                        #identities[item] = (0 , identities[item][1], identities[item][0])
                    elif medias <= identities[item][1][1] and identities[item][0] < -5  and identities[item][2] is not None  and identities[item][2][1] < medias: # <-5
                        status = "¡¡¡ Entrando !!!" if self.regiones[2] == 1 else "¡¡¡ Saliendo !!!"
                        print(f" {self.name}: {status}, {identities[item][0]}")
                        self.result_queue.put(+1*self.regiones[2])
                        #identities[item] = (0, identities[item][1], identities[item][0])
                    
                cv2.imshow(f" prueba {self.name}", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27: #Escape
                    break
            except:
                traceback.print_exc()
                #time.sleep(0.05)

       

class Camera(multiprocessing.Process):

    def __init__(self, task_queue, stream, bytes_f, regiones):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.file_descriptor = stream.raw.fileno()
        self.bytes = bytes_f
        self.region = regiones

    def run(self):
        bg_subs = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=500)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        skip = 0
        count = 0
        while True:
            frame = None
            self.bytes += os.read(self.file_descriptor, 1024)

            a = self.bytes.find(b'\xff\xd8')  # JPEG start
            b = self.bytes.find(b'\xff\xd9')  # JPEG end
            if a != -1 and b != -1:

                jpg = self.bytes[a:b + 2]  # actual image
                self.bytes = self.bytes[b + 2:]  # other informations

                if len(jpg) > 0:
                    if skip == 0:
                        #skip = 1
                        frame = cv2.imdecode(np.fromstring(
                            jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        
                        if np.shape(frame) == ():
                            continue

                        frame = cv2.resize(frame, (640,480))
                        f = frame.copy()
                        
                        info = [
                            ("Frame", count),
                        ]
                        # loop over the info tuples and draw them on our frame
                        for (idx, (k, v)) in enumerate(info):
                            text = "{}: {}".format(k, v)
                            cv2.putText(f, text, (10, frame.shape[0] - ((idx * 20) + 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


                        cv2.imshow(f"camara-{self.name} prueba", f)
                        _ = cv2.waitKey(1) & 0xFF

                        frame_copy = frame.copy()
                        
                      
                        l_superior = int(self.region[1]*1.2) if  int(self.region[1]*1.2) <= 479 else 479
                        l_inferior = int(self.region[0]*.8) if  int(self.region[0]*0.8) >= 0 else 0

                        frame_copy[:l_inferior , :] = np.zeros(frame_copy[:l_inferior, :].shape)
                        frame_copy[l_superior:, :] = np.zeros(frame_copy[l_superior:, :].shape)
                        
    
                        
                        fgMask = bg_subs.apply(frame_copy)

                        
                        

                        _, fgMask = cv2.threshold(fgMask, 130, 250, cv2.THRESH_BINARY)

                        kernel = np.ones((3, 3), np.uint8)
                        fgMask = cv2.erode(fgMask, kernel, iterations=2)


                        # fgMask = cv2.dilate(fgMask,kernel,iterations = 1)
                        cnts, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        #print([cv2.contourArea(c) for c in cnts])
                        enter = len([cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 100]) > 0

                        #mean = cv2.mean(fgMask)[0]
                        
                        count += 1
                        if enter:
                        #if mean > 3.5:
                            if self.task_queue.full():
                                break
                            self.task_queue.put(frame)
                        time.sleep(0.08) #0.1  probar [0.066 (15 fps)]
                    else:
                        #skip -= 1
                        pass

class Counter(multiprocessing.Process):

    def __init__(self, consumers, people_inside, people_queue, people, tabla):
        multiprocessing.Process.__init__(self)
        self.consumers = consumers
        self.people_inside = people_inside
        self.people_queue = people_queue
        self.people = people
        self.tabla = tabla
    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        for q in self.people_inside: 
                q.put(self.people)

        while True:

            while not self.people_queue.empty():
                try:
                    r = self.people_queue.get_nowait()
                    self.people += r
                    print(f"Personas: {self.people}" )
                except:
                    pass
            for q in self.people_inside:
                q.put(self.people)

            if self.people < 0:
                self.people = 0

            send = {
                "info":
                    {
                        "api_key":"000000",
                        "device": self.tabla
                    },
                "data": {"people": self.people}    
            }
            
            try:
                #print(f'http://158.49.112.127:11223/pecera/arduino/json?json={send}'.replace("'", '"'))
                requests.get(f'http://158.49.112.127:11223/insertData?json={send}'.replace("'", '"'))
            except:
                traceback.print_exc()
                pass
            time.sleep(60)
        print("I'm dead")



def main(*argv):

    parser = argparse.ArgumentParser(description='People counter.', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--file', type=str, default="informatica.yaml", help='''Fichero de ejemplo:

camaras:
  - "url1"
  - "url2"
  - "url3"
barreras:
  -  "120, 295"
  -  "170, 320"
  -  "105, 305"
orientacion:
  -  1
  - -1
  - -1
tabla:
  - "UEXCC_INF_PEOPLE"
    ''')

    parser.add_argument('--people', type=int, default=0, help='Número de personas que hay dentro en el momento de ejecutar el script.')


    args = parser.parse_args()
    try:
        file = open(args.file)
    except IOError:
        print("Fichero no encontrado o no accesible, probando con el fichero por defecto")
        try:
            file = open(parser.get_default("file"))
        except IOError:
            print("Fichero no encontrado o no accesible")
            sys.exit(1)
    
    data = load(file, Loader=Loader)
    file.close()

    people = args.people

    streams = [requests.get(x, stream=True) for x in data["camaras"]]
    bytes_s = [b''] * len(streams)
    regiones = [(*tuple(map(int, barreras.replace(" ", "").split(","))), orientacion) for barreras, orientacion in zip(data["barreras"], data["orientacion"])]

    frame_queue = [multiprocessing.Queue() for i in range(len(streams))]
    people_queue = multiprocessing.Queue()
    people_inside_queue = [multiprocessing.Queue() for i in range(len(streams))]


    consumers = [Detector(frame_queue[i], people_queue, people_inside_queue[i], regiones[i]) for i in range(len(streams))]
    _ = [x.start() for x in consumers]

    producers = [Camera(frame_queue[i], streams[i], bytes_s[i], regiones[i]) for i in range(len(streams))]
    _ = [x.start() for x in producers]
    
    counter = Counter(consumers, people_inside_queue, people_queue, people, data["tabla"][0])
    counter.start()

    try:
        signal.pause()
    except KeyboardInterrupt:
        print ("Caught KeyboardInterrupt, terminating workers")


    _ = [x.terminate() for x in producers]
    _ = [x.terminate() for x in consumers]
    counter.terminate()
    cv2.destroyAllWindows()
    print('done')

    sys.exit(0)



main()
