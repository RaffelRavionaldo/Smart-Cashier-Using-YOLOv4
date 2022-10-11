# Smart-Cashier-Using-YOLOv4

# Data Preparing and processing
Untuk mendapatkan data untuk mendeteksi gambar, saya merekam produk di supermarket sehingga data dapat bervariasi (dalam 1 frame video bisa terdapat 1 objek atau lebih)

Dalam proses pengerjaannya, hal yang pertama dilakukan adalah melakukan labelling data terlebih dahulu, pembaca dapat mengetahui cara melakukan labelling pada laman berikut :

https://github.com/tzutalin/labelImg

setelah melakukan labelling, hal selanjutnya yang dilakukan adalah augmentasi gambar

augmentasi yang saya lakukan adalah memutar gambar dan melakukan zoom out dari gambar tersebut, program yang saya dapatkan yang bernama rotate dan zoom out saya dapatkan dari :

https://github.com/usmanr149/Yolo_bbox_manipulation

# Persiapan membuat model YOLOv4
Menginstall CUDA untuk meningkatkan FPS yang didapatkan saat menjalankan program, dengan spesifikasi laptop intel I7-9750H dengan graphic card Nvidia geforce GTX 1650 didapatkan data ~25fps. jika hanya menggunakan CPU didapatkan 3-5 FPS

cara untuk menginstall CUDA dapat dilihat pada document yang bernama : cara menginstall open cv cuda

Program ini tidak sepenuhnya saya buat sendiri, program ini merupakan modifikasi dari program yang dibuat oleh Akash James yang diterbitkan pada web towardsdatascience.com
pembaca dapat mengaksesnya melalui link berikut :

https://towardsdatascience.com/yolov4-with-cuda-powered-opencv-dnn-2fef48ea3984

Untuk mengetahui jarak benda ke kamera, saya menggunakan rumus yang dibuat oleh Pias Paul dkk, pembaca dapat mengakses melalui link berikut :

https://github.com/paul-pias/Object-Detection-and-Distance-Measurement

