from typing import Tuple, Literal
from pathlib import Path
import numpy as np
import os
import struct
from PIL import Image


def grid_graph_signal(frame_shape: Tuple[int, int], mat: str) -> np.ndarray:
    """Convert an image array to a 1-D signal vector and return the original shape.

    Parameters
    - img: np.ndarray representing the image.

    Returns
    - vector: Flattened pixel values.
    - A: Adjacency matrix of the corresponding graph.
    - shape: Original image shape (height, width) for reconstruction.
    """
    m, n = frame_shape
    size = m*n

    M = np.zeros((size, size))
    lap = (mat == 'LAP') 

    for i in range(m):
        for j in range(n):
            pos = i*n + j
            if j < n-1:
                M[pos][pos+1] = M[pos+1][pos] = 1
                M[pos][pos] -= lap
                M[pos+1][pos+1] -= lap
            if i < m-1:
                M[pos][pos+n] = M[pos+n][pos] = 1
                M[pos][pos] -= lap
                M[pos+n][pos+n] -= lap

    M = -M if lap else M

    return M

def hamming_graph_signal(frame_shape: Tuple[int, int], mat: str) -> np.ndarray:
    """Convert an image array to a 1-D signal vector and return the original shape.

    Parameters
    - img: np.ndarray representing the image.

    Returns
    - vector: Flattened pixel values.
    - shape: Original image shape (height, width) for reconstruction.
    """
    m, n = frame_shape
    size = m*n

    lap = (mat == 'LAP')

    M = np.zeros((size, size))

    for i in range(m):
        for j in range(n):
            
            pos = i*n + j
            
            for k in range(1, max(m-i, n-j)):
                i_lin = i+k
                j_lin = j+k
                if i_lin < m:
                    pos_lin = pos + k*n
                    M[pos][pos_lin] = M[pos_lin][pos] = (1/k)
                    M[pos][pos] -= lap*(1/k)
                    M[pos_lin][pos_lin] -= lap*(1/k)
                if j_lin < n:
                    pos_lin = pos + k
                    M[pos][pos_lin] = M[pos_lin][pos] = (1/k)
                    M[pos][pos] -= lap*(1/k)
                    M[pos_lin][pos_lin] -= lap*(1/k)

    M = -M if lap else M

    return M

def get_associated_graph(frame_shape: Tuple[int, int], graph_type: str, mat: str) -> np.ndarray:
    
    match graph_type:
        case 'HAMM':
            return hamming_graph_signal(frame_shape, mat)
        case _:
            return  grid_graph_signal(frame_shape, mat)

def parse_and_segment_image(
        file: str, 
        frame_shape: Tuple[int, int] = (8,8)
    ) -> Tuple[list, np.ndarray, Tuple[int, int]]:
    """Reads the image file into an np.ndarray and segments it into small frames with the given shape.

    Parameters
    - image: Input image array.
    - frame_shape: The shape of each frame for the segmentation process.

    Returns
    - frames: A list containing the data for each frame individually
    - img: The np.ndarray corresponding to the B&W version of the image
    - pad_shape: The shape of the padded image
    """
    cwd = os.getcwd()

    # Lendo e convertendo imagem para escala de cinza
    img = np.asarray(Image.open(f'{cwd}/input/{file}')).astype('float32') / 255
    img = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114

    m,n = img.shape
    r,s = frame_shape

    m_lin = int(np.ceil(m/r))*r
    n_lin = int(np.ceil(n/s))*s

    r_img = np.pad(img, ((0, m_lin-m), (0, n_lin-n)))


    frames = []
    for i in range(0, m_lin, r):
        for j in range(0, n_lin, s):
            frames.append(r_img[i:i+r, j:j+s].flatten())

    return frames, img, (m_lin, n_lin)

def signal_to_sparse_array(vector: "object", file: str) -> "object":
    """Reshape a 1-D signal vector back into an image array with the given shape.

    Parameters
    - vector: Flattened pixel values.
    - shape: Target image shape (height, width).

    Returns
    - img_cmp: Compressed Image.
    - rate: The compression rate 
    """
    k = 0
    size = vector.size
    img_cmp = np.zeros((size, 2))
    i = 0
    while i < size:
        count = 0
        elem = vector[i]
        while i < size and vector[i] == elem:
            count += 1
            i += 1
        img_cmp[k,:] = [elem, count]
        k += 1
    img_cmp = img_cmp[:k]

        
    img_cmp.tofile(f'bin/{file[:-4]}_compressed.bin')

    original_size = os.path.getsize(f'bin/{file[:-4]}.bin')
    compressed_size = os.path.getsize(f'bin/{file[:-4]}_compressed.bin')
    
    return img_cmp, (1 - compressed_size/original_size)

def save_png_file(img_array: np.ndarray, exp_path: Path):

    cmp_name = exp_path.name

    print(f'\n- Exportando imagem descomprimida para {cmp_name}.png!')
      
    img = Image.fromarray(np.clip(img_array*255, 0, 255).astype(np.uint8))
    img.save(f'{exp_path}.png')

def save_img_to_file(
        cmp_img: np.ndarray, 
        or_shape: Tuple[int, int], 
        pad_shape: Tuple[int, int],
        frame_shape: Tuple[int, int],
        graph: str,
        mat: str, 
        exp_path: Path
    ) -> None:

    cmp_name = exp_path.name

    print(f'\n- Exportando imagem comprimida para {cmp_name}.gft!')

    m, n = or_shape
    m_lin, n_lin = pad_shape
    r, s = frame_shape

    # Salvando no formato GFT
    with open(f'{exp_path}.gft', "wb") as f:

        method_mask = (graph=='GRID') + 2*(mat=='ADJ')

        # Armazenando um único byte
        method_info = struct.pack('B', method_mask)
        f.write(method_info)

        x0, x1 = cmp_img.shape

        # Armazenando 24 bytes para os shapes
        shape_info = struct.pack("iiiiiiii", m, n, m_lin, n_lin, r, s, x0, x1)
        f.write(shape_info)

        cmp_img = cmp_img.astype(np.float32)
    
        cmp_img.tofile(f)

        size = cmp_img.nbytes+33

    return size

def read_img_from_file(file_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:

    cwd = os.getcwd()

    with open(file_path, "rb") as f:

        method_info, = struct.unpack('B', f.read(1))
        
        g_val = method_info % 2
        m_val = (method_info >> 1) % 2

        graph = 'GRID' if g_val else 'HAMM'
        mat = 'ADJ' if m_val else 'LAP'

        header_bytes = f.read(32)
        m, n, m_lin, n_lin, r, s, x0, x1 = struct.unpack("iiiiiiii", header_bytes)

        cmp_img = np.fromfile(f, dtype=np.float32)
        cmp_img = cmp_img.reshape((x0, x1))
    
    return cmp_img, (m, n), (m_lin, n_lin), (r, s), graph, mat