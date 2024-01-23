import numpy as np
from sympy import *
import time

class bcolors:
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


#Se a matriz eh definida positiva (aproximado)
def is_pd(x,n):
    if n == 1:
        if x[0] > 0: return 1
    else:
        if np.all(np.linalg.eigvals(x) > 0): return 1

    return 0



#Retorna True se a condicao de Armijo foi satisfeita
def Armijo(f,D,d,p,a): #retorna True se a condicao for satisfeita
    pk = np.add(p, np.multiply(a,d))
    fd = np.matmul(np.transpose(D.subs({x: p[0], y: p[1], z: p[2]})), d)
    return f.subs({x: pk[0], y: pk[1], z: pk[2]}) < f.subs({x: p[0], y: p[1], z: p[2]}) + 0.2 * a * fd


def Metodo_Newton(f,p0,erro,n):

    t = time.time()
    D = derive_by_array(f, (x, y, z))  # gradiente
    H = hessian(f, (x, y, z))  # Hessiana

    p = p0
    i=0

    while True :

        grad = D.subs({x: float(p[0]), y: float(p[1]), z: float(p[2])}) #gradiente no ponto
        grad = np.array(grad,dtype=float)
        grad = grad[:n]


        Hess = H.subs({x: float(p[0]), y: float(p[1]), z: float(p[2])}) #Hessiana no ponto
        Hess = np.array(Hess,dtype=float)
        if n == 1: Hess = [Hess[0][0]]
        if n == 2: Hess = np.delete(Hess[:2], 2, 1)

        if np.linalg.norm(grad) <= erro:  #se gradiente = 0
            print(time.time()-t,"s")

            if is_pd(Hess, n) == 0:
                print("sela")
                return p[:n]

            elif is_pd(Hess, n) == 1:
                print("minimo")
                return p[:n]

        #correcao de dimensao
        if n == 1:
            if Hess[0] == 0: inv = [1]
            else: inv = [1/Hess[0]]
        else:
            if np.linalg.det(Hess) == 0:
                if n==2: inv = [[1,0],[0,1]]
                else: inv = [[1,0,0],[0,1,0],[0,0,1]]

            else : inv = np.linalg.inv(Hess)


        d =  np.matmul(grad,np.negative(inv))
        d = np.array(d, dtype=float)

        #correcao de dimensao
        if n >= 1 and n<3 :d = np.append(d,0)
        if n == 1 : d = np.append(d,0)

        #condicao de armijo
        a=1
        if i != 0: a = max(1, 1 / np.linalg.norm(np.subtract(p, pa))) #passo inicial modificado

        while not Armijo(f,D,d,p,a):
            a=a*0.1
            if a < 10**(-40): return (bcolors.FAIL + "erro, passo=0")

        pa = p
        p = np.add(p, np.multiply(d,a))
        print(p)
        i+=1

    return 0



def Metodo_gradiente(f,p0,erro,n):
    t = time.time()
    D = derive_by_array(f, (x, y, z)) #gradiente
    H = hessian(f, (x, y, z)) #Hessiana

    p=p0
    i=0

    while True :
        grad = D.subs({x:float(p[0]),y:float(p[1]),z:float(p[2])}) #gradiente no ponto p
        grad = np.array(grad, dtype=float)
        Hess = H.subs({x: float(p[0]), y: float(p[1]), z: float(p[2])}) #Hessiana no ponto p
        Hess = np.array(Hess, dtype=float)

        if  np.linalg.norm(grad) <= erro: #se gradiente = 0

            print(time.time()-t,"s")

            #correcao de dimensao
            if n == 1: Hess = [Hess[0][0]]
            if n == 2: Hess = np.delete(Hess[:2], 2, 1)

            if is_pd(Hess, n) == 0:
                print("sela ")
                return p[:n]

            elif is_pd(Hess,n) == 1:
                print("minimo ")
                return p[:n]

        else:
            dir = np.negative(grad)
            #dir = np.array(dir, dtype=float)
            a=1
            if i != 0: a = max(1, 1 / np.linalg.norm(np.subtract(p, pa))) #passo inicial modificado

            while not Armijo(f,D,dir,p,a):
                a = a*0.1
                if a < 10**(-40): return (bcolors.FAIL + "erro, passo=0")

            pa=p
            p = np.add(p, np.multiply(a, dir))
            print(p)
        i+=1

    return 0




x = Symbol("x")
y = Symbol("y")
z = Symbol("z")

erro=10**(-6)

"""""
Exemplos de input

n=1
f = x**2 + 3*x -20 
f = sin(3x + 1)
f = x**3
"""""

"""""
n=2
f=x*y
f=  2**x + 5**y 
f= sin(x)*cos(y) - x
"""""

"""""
n=3
f=(x*y + z)**4
f=1/x + 1/y + 1/z
"""""

n=0
f = 0
p0 = [0,0,0]


print("gradiente : " + bcolors.OKGREEN, Metodo_gradiente(f,p0,erro,n), bcolors.ENDC)
print("---------------------------------------------------------------------------")
print("newton : " + bcolors.OKGREEN, Metodo_Newton(f,p0,erro,n), bcolors.ENDC)
