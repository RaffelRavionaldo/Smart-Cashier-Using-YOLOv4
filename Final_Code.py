import sys
import cv2
import argparse
import random
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

class YOLOv4:
    def __init__(self):
        """ Method called when object of this class is created. """

        self.args = None
        self.net = None
        self.names = None

        self.parse_arguments()
        self.initialize_network()
        self.run_inference()

    def parse_arguments(self):
        """ Method to parse arguments using argparser. """

        parser = argparse.ArgumentParser(description='Object Detection using YOLOv4 and OpenCV4')
        parser.add_argument('--image', type=str, default='', help='Path to use images')
        parser.add_argument('--stream', type=str, default='', help='Path to use video stream')
        # untuk menentukan file config dari YOLOv4 yang akan digunakan
        parser.add_argument('--cfg', type=str, default='yolov4-obj.cfg', help='Path to cfg to use')
        # untuk menentukan file hasil training YOLOv4 yang akan digunakan
        parser.add_argument('--weights', type=str, default='10000.weights', help='Path to weights to use')
        # untuk menentukan file txt yang berisi nama-nama barang yang telah di training
        parser.add_argument('--namesfile', type=str, default='classes.txt', help='Path to names to use')
        parser.add_argument('--input_size', type=int, default=416, help='Input size')
        parser.add_argument('--use_gpu', default=False, action='store_true', help='To use NVIDIA GPU or not')

        self.args = parser.parse_args()

    def initialize_network(self):
        """ Method to initialize and load the model. """

        self.net = cv2.dnn_DetectionModel(self.args.cfg, self.args.weights)
        
        if self.args.use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
        if not self.args.input_size % 32 == 0:
            print('[Error] Invalid input size! Make sure it is a multiple of 32. Exiting..')
            sys.exit(0)
        self.net.setInputSize(self.args.input_size, self.args.input_size)
        self.net.setInputScale(1.0 / 255)
        self.net.setInputSwapRB(True)
        with open(self.args.namesfile, 'rt') as f:
            self.names = f.read().rstrip('\n').split('\n')

    def stream_inf(self):
        """ Method to run inference on a stream. """

        source = cv2.VideoCapture(0 if self.args.stream == 'webcam' else self.args.stream)

        cred = credentials.Certificate('test.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': "https://e-kasa-ips-default-rtdb.firebaseio.com/"
        })

        b = 100
        g = 150
        r = 0

        self.jumlah_ultramilkfc = 0
        self.jumlah_serena = 0

        waktu_awal = time.time()
        while(source.isOpened()):
            ret, frame = source.read()
            if ret:
                timer = time.time()
                classes, confidences, boxes = self.net.detect(frame, confThreshold=0.3, nmsThreshold=0.4)

                waktu_akhir =  time.time()
                #print('[Info] Time Taken: {} | FPS: {}'.format(time.time() - timer, 1/(time.time() - timer)), end='\r')

                if(not len(classes) == 0):
                    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                        label = '%s: %.2f' % (self.names[classId], confidence)
                        left, top, width, height = box
                        b = 100
                        g = 150
                        r = 0
                        cv2.rectangle(frame, box, color=(b), thickness=2)
                        cv2.rectangle(frame, (left, top), (left + len(label) * 20, top - 30), (b, g, r), cv2.FILLED)
                        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (255 - b, 255 - g, 255 - r), 1, cv2.LINE_AA)

                        distancei = (((2 * 3.14 * 180) / (width + height * 360)) * 1000 + 3) *2.54
                        jarak = [self.names[classId],distancei]

                        # Inisialisasi jarak awal
                        if (not len(classes) == 0):
                            if(waktu_akhir - waktu_awal < 8):
                                if classId == 3:
                                    self.jarak_serena_awal =  distancei
                                if classId == 10:
                                    self.jarak_ultra_milk_cream_awal = distancei

                        #perhitungan jarak sekarang
                        if classId == 3:
                            self.jarak_serena_sekarang = distancei
                        if classId == 10:
                            self.jarak_ultra_milk_cream_sekarang = distancei

                        #barang diambil
                        try:
                            if classId == 3:
                                self.ambil_serena()
                                self.letak_serena()
                            if classId == 10:
                                self.ambil_ultramilk()
                                self.letak_ultramilk()
                        except:
                            print("error")

                cv2.imshow('Inference', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def run_inference(self):

        if self.args.image == '' and self.args.stream == '':
            print('[Error] Please provide a valid path for --image or --stream.')
            sys.exit(0)

        elif not self.args.stream == '':
            self.stream_inf()

        cv2.destroyAllWindows()

    def ambil_ultramilk(self):

        if (self.jarak_ultra_milk_cream_awal - self.jarak_ultra_milk_cream_sekarang) > 8:
            ref = db.reference('rak_minuman/minum2/harga')
            harga = ref.get()
            x = ''.join(harga)
            harga = int(x)
            ref = db.reference('user/total_harga/totalharga')
            totalharga = ref.get()
            a = ''.join(totalharga)
            y = int(a)
            total_harga = y + harga
            ref = db.reference('rak_minuman/minum2/name')
            nama = ref.get()

            self.jumlah_ultramilkfc += 1
            ref = db.reference("user")
            ref.update({
                "barang1":
                    {
                        "name": nama,
                        "harga": str(harga),
                        "Jumlahbarang": str(self.jumlah_ultramilkfc)
                    }
            })
            ref = db.reference("user/total_harga")
            ref.update({
                "totalharga": str(total_harga)
            })
            time.sleep(5)
            self.jarak_ultra_milk_cream_awal += 3
            #print(self.jarak_ultra_milk_cream_awal)

    def letak_ultramilk(self):
        if (self.jarak_ultra_milk_cream_sekarang - self.jarak_ultra_milk_cream_awal) > 3:
            #print(self.jarak_ultra_milk_cream_awal)
            ref = db.reference('rak_minuman/minum2/harga')
            harga = ref.get()
            x = ''.join(harga)
            harga = int(x)
            ref = db.reference("user/total_harga/totalharga")
            totalharga = ref.get()
            a = ''.join(totalharga)
            y = int(a)
            total_harga = y - harga
            ref = db.reference("user/total_harga")
            ref.update({
                "totalharga": str(total_harga)
            })
            self.jumlah_ultramilkfc -= 1
            ref = db.reference("user")
            emp_ref = ref.child("barang1")
            emp_ref.update({
                'Jumlahbarang': str(self.jumlah_ultramilkfc)
            })
            time.sleep(5)
            self.jarak_ultra_milk_cream_awal -= 3

    def ambil_serena(self):

        if (self.jarak_serena_awal - self.jarak_serena_sekarang) > 12:
            ref = db.reference('rak_makanan/makanan1/harga')
            harga = ref.get()
            x = ''.join(harga)
            harga = int(x)
            ref = db.reference('user/total_harga/totalharga')
            totalharga = ref.get()
            a = ''.join(totalharga)
            y = int(a)
            total_harga = y + harga
            ref = db.reference('rak_makanan/makanan1/name')
            nama = ref.get()

            self.jumlah_serena += 1
            ref = db.reference("user")
            ref.update({
                "barang3":
                    {
                        "name": nama,
                        "harga": str(harga),
                        "Jumlahbarang": str(self.jumlah_serena)
                    }
            })
            ref = db.reference("user/total_harga")
            ref.update({
                "totalharga": str(total_harga)
            })
            time.sleep(5)
            self.jarak_serena_awal += 3

    def letak_serena(self):

        if (self.jarak_serena_sekarang - self.jarak_serena_awal) > 3:
            print(self.jarak_ultramilkcoklat_awal)
            ref = db.reference('rak_makanan/makanan1/harga')
            harga = ref.get()
            x = ''.join(harga)
            harga = int(x)
            ref = db.reference("user/makanan1/totalharga")
            totalharga = ref.get()
            a = ''.join(totalharga)
            y = int(a)
            total_harga = y - harga
            ref = db.reference("user/total_harga")
            ref.update({
                "totalharga": str(total_harga)
            })
            self.jumlah_serena -= 1
            ref = db.reference("user")
            emp_ref = ref.child("barang2")
            emp_ref.update({
                'Jumlahbarang': str(self.jumlah_serena)
            })
            time.sleep(5)
            self.jarak_serena_awal -= 3

if __name__== '__main__':

    yolo = YOLOv4.__new__(YOLOv4)
    yolo.__init__()