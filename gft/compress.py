from typing import Tuple
import pywt
import numpy as np
import bisect

PRINT_FREQ = 25


def compress_k_greatest_coeffs(coeffs: np.ndarray, k: int, offset: int) -> Tuple[np.ndarray, np.ndarray]:
    """Zero out the low-magnitude coefficients (compress by keeping top-k).

    Parameters
    - coeffs: GFT coefficients.
    - k: Number of largest-magnitude coefficients to retain.

    Returns
    - places: Places of the top-k coefficients.
    - values: Values of the top-k coefficients.
    """
    k = min(k, len(coeffs))
    cmp_signal = np.zeros((2, k))
    best = []
    for i in range(len(coeffs)):
        if len(best) < k:
            bisect.insort(best, (np.abs(coeffs[i]), i))
        elif np.abs(coeffs[i]) > np.abs(best[0][0]):
            del best[0]
            bisect.insort(best, (np.abs(coeffs[i]), i))

    for i in range(k):
        _, pos = best[i]
        cmp_signal[:, i] = [pos+offset, coeffs[pos]]
    
    return cmp_signal
  
def compress_coeffs_under_thr(coeffs: np.ndarray, thr: float, offset: int) -> Tuple[np.ndarray, np.ndarray]:
    """Zero out the low-magnitude coefficients (compress by keeping top-k).

    Parameters
    - coeffs: GFT coefficients.
    - keep_top_k: Number of largest-magnitude coefficients to retain.

    Returns
    - compressed_coeffs: Coefficients after zeroing.
    - num_zeroed: Number of coefficients set to zero.
    """
    n = len(coeffs)
    cmp_signal = np.zeros((2, n))
    k = 0

    for i in range(n):
        if abs(coeffs[i]) >= thr:
            cmp_signal[:, k] = [i+offset, coeffs[i]]
            k += 1
    return cmp_signal[:, :k]

def compress(frames: list, basis: np.ndarray,  method: str, cmp_par: int | float, frame_shape: Tuple[int, int]) -> np.ndarray:
    """Compress frames using GFT basis transformation.
    
    Parameters
    - frames: List of signal frames to compress.
    - basis: Transformation basis matrix.
    - method: Compression method ("THR" for threshold or "KGT" for k-greatest).
    - cmp_par: Compression parameter (threshold value or k count).
    - frame_shape: Shape of each frame (rows, cols).
    
    Returns
    - Compressed signal as array with shape (2, num_coefficients).
    """
    match method:
        case "THR":
            cmp_method = compress_coeffs_under_thr
        case "KGT":
            cmp_method = compress_k_greatest_coeffs
        case _:
            cmp_method =  lambda c, a: c

    r, s = frame_shape
    pixels_per_frame = r*s

    n_frames = len(frames)
    n_pixels = n_frames*pixels_per_frame
    offset = 0

    cmp_signal = np.zeros((2, n_pixels))
    k = 0

    freq = int(PRINT_FREQ*n_frames/100)

    for i, signal in enumerate(frames):

        signal_B = basis.T @ signal

        cmp_signal_B = cmp_method(signal_B, cmp_par, offset)
        
        k_lin = k + cmp_signal_B.shape[1]
        cmp_signal[:, k:k_lin] = cmp_signal_B

        k = k_lin
        offset += pixels_per_frame

        if i % freq == 0:
            print(f'\t{(100*(i+1)/n_frames):.2f}% FINALIZADO')
    
    print('\tFINALIZADO!')
    
    return cmp_signal[:, :k]

def fourier_compression(frames: list, k: int, frame_shape: Tuple[int, int], pad_shape: Tuple[int, int]) -> np.ndarray:
    """Compress a signal using FFT.
    
    Parameters
    - frames: List of signal frames to compress.
    - k: Compression parameter (k count).
    - frame_shape: Shape of each frame (rows, cols).
    
    Returns
    - Compressed signal as array with shape (2, num_coefficients).
    """

    r, s = frame_shape
    m_lin, n_lin = pad_shape

    n_frames = len(frames)

    final_signal = np.zeros(pad_shape)

    freq = int(PRINT_FREQ*n_frames/100)

    frame_rows = int(m_lin//r)
    frame_cols = int(n_lin//s)

    for i in range(frame_rows):
        for j in range(frame_cols):
            pos = i*frame_cols + j

            signal = frames[pos]

            four_base_frame = np.fft.fft(signal)

            idx = np.argsort(np.abs(four_base_frame))[:-k]
            compressed_frame = four_base_frame.copy()
            compressed_frame[idx] = 0

            decmp_frame = (np.fft.ifft(compressed_frame).real).reshape(frame_shape)

            final_signal[i*r : (i+1)*r, j*s: (j+1)*s] = decmp_frame

            if pos % freq == 0:
                print(f'\t{(100*pos/n_frames):.2f}% FINALIZADO')
    
    return final_signal

def haar_compression(frames: list, k: int, frame_shape: Tuple[int, int], pad_shape: Tuple[int, int]):
    """Compress a signal using DWT (Haar).
    
    Parameters
    - frames: List of signal frames to compress.
    - k: Compression parameter (k count).
    - frame_shape: Shape of each frame (rows, cols).
    
    Returns
    - Compressed signal as array with shape (2, num_coefficients).
    """

    r, s = frame_shape
    m_lin, n_lin = pad_shape

    n_frames = len(frames)

    final_signal = np.zeros(pad_shape)

    freq = int(PRINT_FREQ*n_frames/100)

    frame_rows = int(m_lin//r)
    frame_cols = int(n_lin//s)

    for i in range(frame_rows):
        for j in range(frame_cols):

            pos = i*frame_cols + j

            signal = frames[pos]

            coeffs = pywt.wavedec(signal, wavelet='haar')
            coeffs_array, coeff_slices = pywt.coeffs_to_array(coeffs)

            idx = np.argsort(np.abs(coeffs_array))[:-k]
            coeffs_compressed = coeffs_array.copy()
            coeffs_compressed[idx] = 0

            compressed_frame = pywt.array_to_coeffs(
                coeffs_compressed,
                coeff_slices,
                output_format='wavedec'
            )       

            decmp_frame = pywt.waverec(compressed_frame, wavelet='haar').reshape(frame_shape)

            final_signal[i*r:(i+1)*r, j*s:(j+1)*s] = decmp_frame

            if pos % freq == 0:
                print(f'\t{(100*pos/n_frames):.2f}% FINALIZADO')
    
    return final_signal

def decmp_frames_to_img(cmp_signal, basis, pad_shape, or_shape, frame_shape):
    
    r,s = frame_shape
    pixels_per_frame = r*s

    img = np.zeros(pad_shape)
    m, n = pad_shape[0]//r, pad_shape[1]//s

    offset = 0
    size = pad_shape[0]*pad_shape[1]

    freq = int(PRINT_FREQ*size/100)

    for i in range(m):
        for j in range(n):
            
            sig = cmp_signal[offset : offset+pixels_per_frame]

            i_sig = (basis @ sig).reshape(frame_shape)

            img[i*r: (i+1)*r, j*s: (j+1)*s] = i_sig

            offset += pixels_per_frame

            if offset % freq < pixels_per_frame:
                print(f'\t{(100*(offset)/size):.2f}% FINALIZADO')
    
    print('\tFINALIZADO!')

    return img[:or_shape[0], :or_shape[1]]

def decompress(cmp_img: np.ndarray, 
               basis: np.ndarray, 
               pad_shape: Tuple[int, int],
               or_shape: Tuple[int, int],
               frame_shape: Tuple[int, int]
            ) -> np.ndarray:

    size = pad_shape[0]*pad_shape[1]

    decmp_img = np.zeros(size)
    m, n = cmp_img.shape

    for i in range(n):
        pos, val = cmp_img[:, i]
        decmp_img[int(pos)] = val

    
    return decmp_frames_to_img(decmp_img, basis, pad_shape, or_shape, frame_shape)


