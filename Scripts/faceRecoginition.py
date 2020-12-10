import cv2
import os
import numpy as np
import funcao_ativacao as fa

##lembrar de mudar os paths

clf = cv2.CascadeClassifier('C:/Users/grocr/miniconda3/pkgs/libopencv-4.0.1-hbb9e17c_0/Library/etc/haarcascades/haarcascade_frontalface_alt2.xml') #pega o arquivo de reconhecimento facial do opencv. O diretório pode variar dependendo do sistema operacional
##ajustar para reconhecimento dos olhos dentro do retângulo -> se sim tira a foto, se não, deixa quieto

def neural(image):

    ## Definindo as funções principais

    path_bias ='C:/Users/grocr/OneDrive/Área de Trabalho/IA/IA-do-Ramo/Promissores/Bias-NNN-SIG-MSE-0.5-3000-0.3' #diretório bias
    path_pesos = 'C:/Users/grocr/OneDrive/Área de Trabalho/IA/IA-do-Ramo/Promissores/Pesos-NNN-SIG-MSE-0.5-3000-0.3' #diretório pesos

    ##

    def __init__(self): #por conta da alteração no funcionamento da rede eu tive que criar esse construtor
        self.image = image

    ##

    def process_entradas(self): #processa as imagens
        img = cv2.imread(image, 0)
        img = cv2.resize(img,(28,28)) / 255
        return np.reshape(img,784)

    ##

    def read_weights_bias(path_bias, path_pesos, div): ##lê os pesos e as bias de determinado diretório
        
        w_layer1 = np.genfromtxt(path_pesos +'/Pesos1.txt', delimiter= ',')
        w_layer2 = np.genfromtxt(path_pesos +'/Pesos2.txt', delimiter=',')
        w_layer3 = np.genfromtxt(path_pesos +'/Pesos3.txt', delimiter=',')
        w_layer4 = np.genfromtxt(path_pesos +'/Pesos4.txt', delimiter=',')
        
        b_layer1 = np.genfromtxt(path_bias +'/Bias1.txt', delimiter= ',')
        b_layer2 = np.genfromtxt(path_bias +'/Bias2.txt', delimiter=',')
        b_layer3 = np.genfromtxt(path_bias +'/Bias3.txt', delimiter=',')
        b_layer4 = np.genfromtxt(path_bias +'/Bias4.txt', delimiter=',')
        
        return w_layer1.reshape((784,53//div)), w_layer2.reshape((53//div,36//div)), w_layer3.reshape((36//div,25//div)), w_layer4.reshape((25//div,1)), b_layer1.reshape((1,53//div)), b_layer2.reshape((1,36//div)), b_layer3.reshape((1,25//div)), b_layer4.reshape((1,1))
    
    ##

    def predict (test,funcao_ativacao): #faz os calculos de determinada entrada

        if funcao_ativacao == "sigmoid":
            f = fa.sigmoid
        elif funcao_ativacao == "relu":
            f = fa.relu
        elif funcao_ativacao == "leaky_relu":
            f = fa.leaky_relu

        camada_oculta1 = f(np.dot(test, w_layer1) + b_layer1)
        camada_oculta2 = f(np.dot(camada_oculta1, w_layer2) + b_layer2)
        camada_oculta3 = f(np.dot(camada_oculta2, w_layer3) + b_layer3)

        return f(np.dot(camada_oculta3,w_layer4) + b_layer4)
    
    ##

    ## Chamando as funções

    w_layer1, w_layer2, w_layer3, w_layer4, b_layer1, b_layer2, b_layer3, b_layer4 = read_weights_bias(path_bias, path_pesos, 1)

    img = process_entradas('./faces')

    return predict(img,"sigmoid")


def faceRecoginition():
    cont = 0
    if(os.path.isdir('./faces') == False): #verificação. Existe uma pasta com esse nome? 
        os.mkdir('./faces') #caso não exista ela é criada

    cam = cv2.VideoCapture(0) #captura a câmera do computador/notebook
    while (not cv2.waitKey(115) & 0xFF == ord('s')): #exibe a câmera em tempo real até que a tecla "s" seja apertada
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #conversão em cinza para a detecção do rosto
        faces = clf.detectMultiScale(gray) #faces é uma matriz com as coordenadas em pixel de cada rosto
        for x, y, w, h in faces:
            image = frame[y-60: y+h+100, x-100: x+w+100]
            cv2.imwrite('./faces/face{}.png'.format(cont), image)
            if (neural('./faces/face{}.png'.format(cont)) < 0.5):
                print(neural('./faces/face{}.png'.format(cont)))
                cv2.rectangle(frame, (x,y), (x+w,y+h), (152,56,255))
            else:
                print(neural('./faces/face{}.png'.format(cont)))
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,111,0))
            cv2.imshow('video', frame) #mostra o vídeo
            cont += 1
    cam.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows() #fecha as janelas


faceRecoginition()
