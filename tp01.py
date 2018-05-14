import tensorflow as tf
import numpy as np
from numpy import genfromtxt

sess = tf.InteractiveSession()

int_classes   = []
vetor_numeros = []
vetor_classes = []

### HIPERPARAMETROS ###
taxa_aprendizado       = 0.5                #alterada de 0.5, 1.0 e 10
unidades_camada_oculta = 25                 #alterada de 25, 50 e 100
numero_epocas          = 100
batch_size             = 1                  #Gradiente Descendente = 5000 // Gradiente Estocastico = 1 // Mini-Batch = 10 ou 50

X = tf.placeholder(tf.float32, shape=[None, 784], name='DadoEntrada') #shape da mnist 28*28=784
Y = tf.placeholder(tf.float32, shape=[None, 10],  name='DadoSaida')   #10 classes de saida, 0 a 9 

#Peso(W) e vies(b) para a camada oculta
W1 = tf.Variable(tf.random_normal([784, unidades_camada_oculta]), dtype=tf.float32, name='Weights')
b1 = tf.Variable(tf.random_normal([unidades_camada_oculta]), dtype=tf.float32, name='biases')

#Peso(W) e vies(b) para a camada de saida
W2 = tf.Variable(tf.random_normal([unidades_camada_oculta, 10]), dtype=tf.float32, name='Weights')
b2 = tf.Variable(tf.random_normal([10]),                         dtype=tf.float32, name='biases')

### LEITURA DE DADOS ###
arquivo = genfromtxt('./data_tp1', delimiter=',', autostrip=True, dtype=int)

for x in range(1, 4999):
    linha  = np.array(arquivo[x-1:x])       #np array 'linha' contendo a linha (785 itens)
    classe = np.resize(linha, (1,1))        #aqui, pegamos o primeiro valor de 'linha'
    
    linha_escalar = linha.ravel()           #escalar da linha com os 785 itens
    numero = linha_escalar[1:785:1]         #escalar do numero, sem o primeiro item (784 itens)

    int_classes.append(int(classe))         #adiciono a classe no vetor de classes
    vetor_numeros.append(np.array(numero))  #adiciono o numero no vetor de numeros

### CONVERTENDO CLASSES EM VETORES ONE-HOT ###

#Para cada inteiro do vetor 'int_classes', crio um array de 10 posicoes representando os numeros de 0 a 9, com 0 em todas casas, menos na que representa o numero daquela classe, que tera o numero 1
vetor_classes = np.eye(10, dtype=int)[int_classes]  ###

### MODELO DE REDE ####
def mnist(X):
    h1           = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    camada_saida = tf.matmul(h1, W2) + b2
    return camada_saida

#Construindo modelo de rede
logits = mnist(X)

funcao_perda = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
otimizador_gradiente_descendente = tf.train.GradientDescentOptimizer(taxa_aprendizado).minimize(funcao_perda)

### INICIALIZAR VARIAVEIS ###
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)                          #chama o inicializador das variaveis
    
    ### TREINO ###
    for epoca in range(numero_epocas):
        inicio      = 0
        fim         = 0
        total_batch = int(5000/batch_size)

        #for i in range(total_batch):    
        while inicio < len(vetor_numeros):
            fim   += batch_size
            batch  = []
            
            numero = np.asarray(vetor_numeros[inicio:fim])
            batch.append(numero)
           
            classe = np.asarray(vetor_classes[inicio:fim])
            batch.append(classe)

            otimizador_gradiente_descendente.run(feed_dict={X: batch[0], Y:batch[1]})
            inicio += batch_size

    ### TESTE DO MODELO ###
    predicao_modelo = tf.nn.softmax(logits)
    predicao_correta = tf.equal(tf.argmax(predicao_modelo, 1), tf.argmax(Y, 1))

    ### CALCULO DA ACURACIA ###
    perda = funcao_perda.eval(feed_dict={X: vetor_numeros, Y: vetor_classes})
    acuracia = tf.reduce_mean(tf.cast(predicao_correta, tf.float32))
    print("Epoca: %d, Perda: %.f" % (epoca+1, perda))
    print("Acuracia obtida: ", 100*acuracia.eval(feed_dict={X: vetor_numeros, Y: vetor_classes}))

