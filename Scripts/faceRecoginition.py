import cv2
import numpy as np

##lembrar de mudar os paths

clf = cv2.CascadeClassifier('C:\\haarcascade_frontalface_default.xml') #pega o arquivo de reconhecimento facial do opencv. O diretório pode variar dependendo do sistema operacional
##ajustar para reconhecimento dos olhos dentro do retângulo -> se sim tira a foto, se não, deixa quieto
path_bias ='C:\\Users\\natan\\OneDrive\\Área de Trabalho\\IA-do-Ramo\\P&B-FINAL\\Bias-BCE-FINAL' 
path_pesos = 'C:\\Users\\natan\\OneDrive\\Área de Trabalho\\IA-do-Ramo\\P&B-FINAL\\Pesos-BCE-FINAL'

def sigmoid (x):
    return 1.0/ (1.0 + np.exp(-x))

def read_weights_bias(path_bias, path_pesos, div):
    
    w_layer1 = np.genfromtxt(path_pesos +'/Pesos1.txt', delimiter= ',')
    w_layer2 = np.genfromtxt(path_pesos +'/Pesos2.txt', delimiter=',')
    w_layer3 = np.genfromtxt(path_pesos +'/Pesos3.txt', delimiter=',')
    w_layer4 = np.genfromtxt(path_pesos +'/Pesos4.txt', delimiter=',')
    
    b_layer1 = np.genfromtxt(path_bias +'/Bias1.txt', delimiter= ',')
    b_layer2 = np.genfromtxt(path_bias +'/Bias2.txt', delimiter=',')
    b_layer3 = np.genfromtxt(path_bias +'/Bias3.txt', delimiter=',')
    b_layer4 = np.genfromtxt(path_bias +'/Bias4.txt', delimiter=',')
    
    return w_layer1.reshape((784,53//div)), w_layer2.reshape((53//div,36//div)), w_layer3.reshape((36//div,25//div)), w_layer4.reshape((25//div,1)), b_layer1.reshape((1,53//div)), b_layer2.reshape((1,36//div)), b_layer3.reshape((1,25//div)), b_layer4.reshape((1,1))


def faceRecoginition():
    faces = []
    cam = cv2.VideoCapture(0) #captura a câmera do computador/notebook
    while (not cv2.waitKey(115) & 0xFF == ord('s')): #exibe a câmera em tempo real até que a tecla "s" seja apertada
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #conversão em cinza para a detecção do rosto
        try:
            faces = clf.detectMultiScale(gray) #faces é uma matriz com as coordenadas em pixel de cada rosto
            for x, y, w, h in faces:
                image = gray[y-60: y+h+100, x-100: x+w+100]
                image = cv2.resize(image,(28,28)) / 255
                image = image.flatten().reshape((1,784))
                camada_oculta1 = sigmoid(np.dot(image,w_layer1) + b_layer1)
                camada_oculta2 = sigmoid(np.dot(camada_oculta1,w_layer2) + b_layer2)
                camada_oculta3 = sigmoid(np.dot(camada_oculta2,w_layer3) + b_layer3)
                val = sigmoid(np.dot(camada_oculta3,w_layer4) + b_layer4)[0]
                if (val < 0.5):
                    print(val)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
                else:
                    print(val)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0))
                cv2.imshow('video', frame) #mostra o vídeo
        except:
            pass
    cam.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows() #fecha as janelas

w_layer1, w_layer2, w_layer3, w_layer4, b_layer1, b_layer2, b_layer3, b_layer4 = read_weights_bias(path_bias, path_pesos, 1)

faceRecoginition()
