# CURSO DE MUSICA

# Tema: Descripcion del sonido y musica
Para describir el sonido con metodos simples de Machine Learning se hara uso de una API "Freesound" para cargar descriptores de sonido precomputados y desarrollar agrupacion y clasificacion de sonidos.

Este tema consistira en cuatro partes: 1) Descarga de sonidos y descriptores desde ***Freesound***, 2) Seleccionar dos descriptores para una buena agrupar de sonido, 3) Agrupar sonidos usando ***k-means***, y 4) Clasificar sonidos usando ***k-NN***.

### Conceptos Relevantes

#### Freesound API
Con la API de ***Freesound*** nosotros podremos navegar, buscar y recuperar informacion desde ***Freesound***. Nosotros tambien podemos desarrollar queries avanzados combinando contenido de analisis de caracteristicas otros metadatas (etiquetas, etc ...). Con la API podemos hacer una busqueda de texto similar a como hacemos en buscador avanzado en el sitio https://freesound.org/search/?q, excepto implentar los queries en software.

#### Sound descriptors
En este ejercicio, usaremos descriptores de sonidos que han sido pre-computados con ***Essentia***, https://essentia.upf.edu y son almacenados en la base de datos de Freesound junto con el correspondiente sonido. Muchos descriptores de sonidos pueden ser extraidos usando Essentia y en Freesound, un numero de ellos son usados.

#### Distancia Euclidiana
La distancia Euclidiana es la distancia de una linea recta entre dos puntos en un espacio n-dimensional, asi la distancia entre dos puntos $p$ y $q$ es la longitud del segmento de linea que los conecta. Si $p = (p_1, p_2,...,p_n)$ y $q = (q_1, q_2,...,q_n)$ son dos puntos en el n-espacio Euclidiano, luego la distancia, $d$, desde $p$ a $q$, o desde $q$ a $p$ esta dado por la formula de pitagoras:

$d(p,q) = \sqrt{\sum^n_{i=1} (q_i - p_i)^2}$

### Agrupamiento k-means (k-means)
El agrupamiento ***k-means*** es un metodo de cuantificacion vectorial que es popular en el analisis de agrupamiento en la mineria de datos.
el agrupamiento ***k-means*** tiene como objetivo dividir en $n$ observaciones dentro de $k$ grupos in el cual cada observacion pertenece a un grupo con el significado mas cercano, sirviendo como un prototipo de grupos. el problema es computacionalmente dificil (NP-hard), sin embargo, eficientes algoritmos euristicos convergen rapidamente local optimo.

Dado un conjunto de observaciones $(x_1, x_2, ..., x_n)$, donde cada observacion es un vector real d-dimesional, el agrupamiento k-means tiene como objetivo dividir las $n$ observaciones en $k (<= n)$ conjuntos $S = {S_1, S_2, ..., S_k}$ asi minimizar la suma de cuadrados de los clusters internos, el objetivo es encontrar:

$\underset{\mathbf{S}} {\operatorname{arg\,min}}  \sum_  {i=1}^{k} \sum_{\mathbf x \in S_i} \left\| \mathbf x - \boldsymbol\mu_i \right\|^2 = \underset{\mathbf{S}} {\operatorname{arg\,min}}  \sum_  {i=1}^k |S_i| \operatorname{Var} S_i$, donde $μ_i$ es el significado de los puntos $S_i$.

# Desarrollo del ejercicio
## Parte 1: Descarga de sonidos y descriptores desde Freesound
Descargar una coleccion de sonidos de intrumentos y sus descriptores desde Freesound unsando la API Freesound.

**Primero.-** Debemos de obtener una llave de la API Freesound (Freesound API key) para hacer uso de esta. En caso que no tengamos una cuenta de Freesound debemos crear uno, para ello ingresamos a la siguiente link https://www.freesound.org/apiv2/apply/. Una vez creado una cuenta nuevamente ingresamos al link anterior, si es la priemera vez que hacemos uso de esta API, nos pedira generar una ***API key*** para lo cual nos pedira ciertos datos del proyecto en el cual estamos trabajando. Cumplido con lo pedido se generara el ***API key***.

**Segundo.-** Dentro del directorio (nuestro espacio de trabajo) en el cual se encuentre nuesto notebook "00.Sound-and-music-description.ipynb" creamos dos  directorios con los nombres `testDownload` y `oneSound` para almacenar los sonidos y descriptores.

**Tercero.-** En seguida debemos instalar el cliente python para Freesound API. Parra ello debemos clonar el repositorio https://github.com/MTG/freesound-python, dentro de nuestro espacio de trabajo. Al clonar el repositorio, dentro de nuestro espacio de trabajo se creara un directorio `freesound-pytho`. La estructura de nuestro espacio de trabajo sera la siguiente:

     |-- workspace (nuestro espacio de trabajo)
        |-- 0.0Sound-and-music-description.ipynb
        |-- testDownload/
        |-- oneSound/
        |-- freesound-python/

**Cuarto.-** Ingresamos dentro de la carpeta `freesound-python` y ejecutamos el siguiente comando.

>*$ python setup.py install*

El comando anterior instalara todas las dependencias de freesound dentro de nuestro ambiente.

**Quinto.-** Ahora ya podremos hacer llamado a la funcion `download-sounds-freesound()` el cual tiene los siguientes parametros:

1. `queryText` (string): Es una simple palabra una cadena de palabras sin espacios (hacer uso de guiones), es tipicamente el nombre del instrumento de cual se quiere obtener el sonido. Ejemplo (eg. "violin", "trumpet", "cello", "bassoon", etc.)
2. `tag` (string): Etiqueta para ser usado en el filtrado del sonido buscado. Ejemplo ("multisample", "sigle-note", "velocity", "tenuto", etc.)
3. `duration` (tupla de 2 numeros de punto flotante): minimo y maxima duracion (segundos) del sonido para filtrar. Ejemplo (0.2, 15).
4. `API_Key` (string): nuestra API key el cual fue optenido de https://freesound.org/apiv2/apply/ y tiene la siguiente estructura "7nCLWEwre4wsjWbZ0wDrersMCey48M7bj128cX3s"
5. `outputDir` (string): Ruta del directorio donde queremos que se almacene los sonidos y sus descriptores descargados, esta ruta puede ser absoluta o relativa; ser recomienda una ruta relativa respecto  nuestro espacio de trabajo en este caso podran ser *./testDownload* o *./oneSound*
6. `topNResults` (integer): Numero de resultados (sonidos) que queremos descargar del instrumento elegido.
7. `featureExt` (file extension): Extension de archivo para los sonidos y descriptores almacenados (.json) generalmente usado.

Para llamar a la funcion `download_sound_freesound()` tendremos que escoger el queryText, tag, y duracion, para que nos retorne una nota singular de los sonidos del instrumento. Los 20 primeros resultados del query deben ser "buenos". Notar que tag puede estar vacio. Por ejemplo para obtener una nota singular de un violin podemos ejecutar la funcion de la siguiente menera:

>`download_sound_freesound(queryText="violin", API_Key=<tu key>, outputDir="./testDownload", topNResults=20, duration=(0, 8.5), tag="single-note")`

Lo anterior retornara 20 notas sigulares de sonidos de violin y los scripts son almacenados en la ruta `./testDownload`. Para este ejercicio debemos descargar 3 sonidos, para ello podemos elegir algunos de estos intrumentos (violin, guitar, bassoon, trumpet, clarinet, cello, naobo).

## Parte 2:Seleccione dos descriptores para un buen agrupamiento de sonido

**Primero.-** Instalamos dependecias que nos permitiran graficar con python, para ello instalamos las librarias en nuestro ambiente virtual:

* numpy
* matplotlib
* scipy

`$ conda install numpy matplotlib scipy`

**Segundo.-** Para el uso de los modulos haremos uso del directorio `testDownload`, el cual le pasaremos como argumento del parametro (inputDir) y el par de indices del descriptor (descInput). Asegurarnos que solo debe haber 3 sonidos de intrumentos dentro del directorio `testDownload`.

**Tercero.-** Eleccion de un buen par de descriptores para los sonidos que fueron descargados en parte 1. Un buen par de descriptores conduce a puntos de distribucion donde todos los sonidos de un instrumento se agrupan juntos, con una buena separacion de los otros agrupamientos de instrumentos. Intentar diferentes combinaciones de pares de descriptores.

**Cuarto.-** Dentro del codigo modificamos las variables `inputDir` y `descInput`, segun a la ruta de nuestro directorio donde se encuntra los sonidos descargados, y las el par de conbinaciones de descriptores respectivamente.

`inputDir = "./testDownload"`

`descInput = (3, 6)`

**Quinto.-** Analizar la grafica y ver con que par de descriptores se genera un mejor agrupamiento de los sonidos.

## Parte 3: Agrupamiento de sonidos usando K-means


## Parte 4: Clacificacion de sonidos con k-NN
