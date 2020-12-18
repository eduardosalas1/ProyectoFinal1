# Base de Datos - Proyecto 3 - Team Los Peloteros

# Reconocimiento facial

La biblioteca de reconocimiento facial para python es el pilar de este proyecto, utilizando solo sus funciones básicas podemos transformar una imagen con una cara en su vector característico. Usamos este vector para comparar los rasgos faciales y determinar si la misma persona aparece en ambas imágenes o si al menos son iguales.

# Preprocesamiento

Dada una base de datos con varias imágenes de personas conocidas, necesitamos obtener cada vector para cada imagen y asociar el nombre correspondiente para cada vector. Luego, escribimos la lista resultante de listas en el disco para que solo hagamos esta operación una vez.


# Algoritmo KNN
KNN es un método utilizado para la clasificación y las tareas de regresión. En este proyecto cumplirá el papel de
un clasificador. Los K vecinos más cercanos deben devolver puntos (o caras en este caso) que son similares
en el camino de la categoría de la primera. Aquí comparamos los tiempos que tarda una base de datos de tamaño variable N para buscar el KNN. En nuestros ejemplos usaremos K = 10, o las 10 caras más similares. Las pruebas de velocidad se realizan en un KNN implementado secuencialmente y luego el uso de una estructura de datos de indexación multidimensional de soporte llamada RTree. Aquí comparamos son eficientes.

## Secuencial y su implementación

Debemos ordenar la lista de vectores característicos usando la distancia hacia los inputs como llave. Así, el algoritmo secuencial puede encontrar los *k* rostros más cercanos en toda la base de datos que se parezcan al input. Después se toman los primeros *k* elementos de la lista resultante y podemos optar entre distancia Euclideana o Manhattan.

## RTree y su implementación
Usaremos la estructura de datos multidimencional Rtree porque es muy útil para contenidos espaciales. En este trabajo, un vector es un punto en un espacio dimensional *x* donde *x* es la longitud del vector. El Rtree será implementado gracias a la libreria de python llamada *rtree*.

Para la inicializacion, se leen los vectores resultantes y se crea un Rtree con un índice multidimensional de 128 dimensiones. Luego se procede a insertar cada vector como una *región* del área dentro del árbol. Esto se realiza para los valores de N (100, 200, ..., 12800). Finalmente se ejecuta la consulta, la cual es un vector aleatorio que se escoge previamente.

## Experimentos
Para los distintos valores de *N* que representan el tamaño de la coleccióm, tomamos el tiempo del rendimiento de ambas estructuras. Usamos KNN search con *k* = 16 para cada *N*:

| Time        | KNN-RTree | KNN-Sequential |
| ----------- | --------- | -------------- |
| *N* = 100   | 0.8595    | 0.8906         |
| *N* = 200   | 0.8362    | 0.8077         |
| *N* = 400   | 0.8718    | 0.8221         |
| *N* = 800   | 0.8446    | 0.9090         |
| *N* = 1600  | 0.8768    | 0.8787         |
| *N* = 3200  | 1.0006    | 0.9886         |
| *N* = 6400  | 1.1493    | 1.1610         |
| *N* = 12800 | 1.5023    | 1.4575         |

