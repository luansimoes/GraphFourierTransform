from typing import Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import itertools
import pandas as pd
from datetime import datetime
import numpy as np

parent_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
output_dir = parent_dir/'output'
input_dir = parent_dir/'input'
sys.path.append(str(parent_dir))
sys.path.append(str(current_dir))

from compress import compress, decompress, fourier_compression, haar_compression
from parse import parse_and_segment_image, save_img_to_file, read_img_from_file, get_associated_graph, save_png_file
from interface import get_main_menu_option, get_single_file_config, get_show_file_config, get_compress_all_config, get_report_config

EXP_TYPE = {
        'THR' : float,
        'KGT' : int,
    }

DEFAULT = {
        'THR' : 1e-2,
        'KGT' : 10
    }

def parse_args(config: dict, single_file: bool = True) -> None:
    """Parse command-line arguments and return a config dictionary.

    Returns
    - config: Dict with options like img_name, cmp_method, cmp_par.
    """

    def check_valid_choice(arg_name: str, arg_ops: tuple) -> None:
        if config[arg_name].upper() not in arg_ops:
            print(f'\nAVISO: {config[arg_name]} não é um comando conhecido para o grafo de representação {arg_ops}. Será utilizado o padrão {arg_ops[0]}.')
            config[arg_name] = arg_ops[0]
        else:
            config[arg_name] = config[arg_name].upper()


    if single_file and not (parent_dir / f'input/{config['img_file']}').is_file():
        print('ERRO: ARQUIVO DE ENTRADA NÃO CONSTA NA PASTA DE INPUT!!')
        exit(1)

    check_valid_choice('graph', ('HAMM', 'GRID'))
    check_valid_choice('graph_mat', ('ADJ', 'LAP'))
    check_valid_choice('cmp_method', ('THR', 'KGT'))


    try:
        frame_shape = config['frame_shape'].strip().split()
        match len(frame_shape):
            case 2:
                frame_shape = (int(frame_shape[0]), int(frame_shape[1]))
            case 1:
                n = int(frame_shape[0])
                print(f'ATENÇÃO: Subimagens serão quadradas {n}x{n}.')
                frame_shape = (n, n)
            case _:
                print(f'ATENÇÃO: Dimensões inválidas para subimagens. Será usado o padrão 8x8.')
                frame_shape = (8, 8)

    except ValueError:
        print(f'ATENÇÃO: Dimensões {tuple(frame_shape)} inválidas para subimagens. Será usado o padrão 8x8.')
        frame_shape = (8, 8)

    finally:
        config['frame_shape'] = frame_shape


    cmp = config['cmp_method']
    try:
        config['cmp_par'] = EXP_TYPE[cmp](config['cmp_par'])
    except ValueError:
        print(f'ATENÇÃO: Método {cmp} espera parâmetro de tipo {EXP_TYPE[cmp]}. Será usado o valor default {DEFAULT[cmp]}.')
        config['cmp_par'] = DEFAULT[cmp]

def show_comparison(or_img: np.ndarray, final_img: np.ndarray) -> None:
    '''Plot original image and final image, for comparison purposes.'''
    plt.figure(num="ORIGINAL")
    plt.imshow(or_img, cmap='gray')
    plt.axis('off') 

    plt.figure(num="COMPRIMIDA")
    plt.imshow(final_img, cmap='gray')
    plt.axis('off')

    plt.show()

def run_compression_pipeline(
        img_file: str, 
        graph: str, 
        graph_mat: str, 
        frame_shape: Tuple[int, int], 
        cmp_method: str, 
        cmp_par: int | float,
        exp_path: Path, 
        M: np.ndarray = None,
        show: bool = True
    ) -> Tuple[float, float]:
    """Main pipeline for compression. read image -> build graph -> decompose -> transform -> reconstruct.

    Parameters
    - img_file: The name of the input file.
    - graph: Configuration for the associated graph (HAMM or GRID).
    - graph_mat: The chosen representation for the graph (ADJ or LAP).
    - frame_shape: The dimensions for the frame segmentation.
    - cmp_method: The compression strategy used to select coefficients.
    - cmp_par: Parameter value for the chosen compression method.
    - exp_path: Path where compressed and reconstructed outputs are saved.
    - M: Optional precomputed graph matrix to reuse across runs.
    - show: Whether to display the original and decompressed images.

    Returns
    - error: Relative L2 error.
    - mse: Mean squared error between original and reconstructed image.
    - compression_rate: Compression ratio in terms of retained coefficients.
    - size_change: Tuple with original and compressed file sizes in megabytes.

    """
    print('\n', '-'*10, f'INICIANDO PROCESSO DE COMPRESSÃO PARA ARQUIVO {img_file}', '-'*10)

    print('\n- SEGMENTANDO IMAGEM E CONSTRUINDO GRAFO ASSOCIADO')

    # Separa a imagem em pequenos quadros e guarda o shape original, o modificado e o grafo correspondente
    frames, img, pad_shape = parse_and_segment_image(img_file, frame_shape)
    M = get_associated_graph(frame_shape, graph, graph_mat) if M is None else M

    or_shape = img.shape

    print('\n- INICIANDO COMPRESSÃO PARA CADA FRAME:')

    e_val, eigenbasis = np.linalg.eigh(M)

    # Comprime a imagem descartando os coeficientes baixos e guarda o número de frames 
    cmp_signal = compress(frames, eigenbasis, cmp_method, cmp_par, frame_shape)

    print('\n- EXPORTANDO ARQUIVO COMPRIMIDO')
    cmp_size = save_img_to_file(cmp_signal, or_shape, pad_shape, frame_shape, graph, graph_mat, exp_path)

    compression_rate = (1 - cmp_signal.shape[1]/(pad_shape[0]*pad_shape[1]))
    size_change = img.nbytes*1e-6, cmp_size*1e-6

    print('\n- REALIZANDO DESCOMPRESSÃO E RECONSTITUINDO IMAGEM')
    final_img = decompress(cmp_signal, eigenbasis, pad_shape, or_shape, frame_shape)
    
    save_png_file(final_img, exp_path)

    error = np.linalg.norm(img - final_img) / np.linalg.norm(img)
    mse = np.mean((img - final_img) ** 2)

    print(f'\n\nTAXA DE COMPRESSÃO (COEFFS): {100*compression_rate:.2f}%')
    print(f'ALTERAÇÃO NO TAMANHO (DO ARRAY):\t{size_change[0]:.2f}MB\t===>\t{size_change[1]:.2f}MB')
    print(f'PERCENTUAL DE ERRO: {100*error:.2f}%')
    print(f'ERRO MÉDIO QUADRÁTICO: {mse:.4f}')

    if show:  
        show_comparison(img, final_img)

    return error, mse, compression_rate, size_change

def single_file_compression() -> None:
    '''Run the pipeline for single-file compression.'''

    # Criando pasta SINGLE_FILE caso ainda não exista
    (output_dir/'SINGLE_FILE').mkdir(parents=True, exist_ok=True)

    # Solicitando e tratando dados de entrada
    config = get_single_file_config()
    config['exp_path'] = output_dir / f'SINGLE_FILE/{config['exp_path']}'
    parse_args(config)
    
    # Executando o pipeline de compressão
    run_compression_pipeline(**config)

def compress_all() -> None:
    '''Run the pipeline for compressing all files in the input directory.'''
    
    config = get_compress_all_config()
    parse_args(config, single_file=False)

    exp_path = output_dir / f'{config['graph']}_{config['graph_mat']}_{config['cmp_method']}_{config['frame_shape']}_{config['cmp_par']}'
    (exp_path).mkdir(parents=True, exist_ok=True)

    all_files = [f for f in input_dir.rglob("*.png")]

    M = get_associated_graph(config['frame_shape'], config['graph'], config['graph_mat'])

    n = len(all_files)

    for i, f in enumerate(all_files):
        print(f'\n\n[{i+1}/{n}] Arquivo {f.name}')
        run_compression_pipeline(img_file=f.name, exp_path=exp_path/(f.name[:-4]), M=M, **config)

def generate_full_report() -> None:
    '''Run the pipeline for generating full reports for the input images.'''

    all_files = [f for f in input_dir.rglob("*.png")]

    report_type = get_report_config()
    if report_type.isdigit() and 1 <= int(report_type) <= 2:
        report_type = int(report_type)
    else:
        report_type = 1
    
    match report_type:
        case 1:
            graphs = ['HAMM', 'GRID']
            mats = ['ADJ', 'LAP']
            frame_shapes = [(4, 4), (8, 8), (16, 16), (32, 32)]
            cmp_rates = [0.1]
        case 2:
            graphs = ['GRID']
            mats = ['LAP']
            frame_shapes = [(32, 128)]
            cmp_rates = [0.02, 0.05, 0.1, 0.2]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_path = parent_dir/'reports'/f'R{report_type} ({timestamp})'

    base_path.mkdir(parents=True, exist_ok=True)
    (base_path/'images').mkdir(parents=True, exist_ok=True)

    exec_graphs = []
    exec_mats = []
    exec_shapes = []
    exec_cmp_rates = []
    exec_mean_errors = []
    exec_mse_errors = []

    for g, mat, frame_shape, cmp_rate in itertools.product(graphs, mats, frame_shapes, cmp_rates):

        print('\n', '-'*10, f'Parâmetros {g}_{mat}_{frame_shape}_{cmp_rate}', '-'*10)
        
        M = get_associated_graph(frame_shape, g, mat)
        
        config = {
            'graph': g,
            'graph_mat': mat,
            'frame_shape': frame_shape,
            'cmp_method': 'KGT',
            'cmp_par': int(cmp_rate*frame_shape[0]*frame_shape[1])
        }

        n = len(all_files)

        # Inicializa colunas do csv que será gerado
        img_names = []
        error_values = []
        mse_values = []

        for i, f in enumerate(all_files):

            name = f.name[:-4]

            print(f'\n\n[{i+1}/{n}] Arquivo {f.name}')
            error, mse,  _, _ = run_compression_pipeline(
                img_file=f.name, 
                exp_path=base_path/f'images/{name}', 
                M=M, 
                show=False, 
                **config
            )

            img_names.append(name)
            error_values.append(round(error, 4))
            mse_values.append(round(mse, 4))
        
        per_img_data = {
            'img_name' : img_names,
            'error' : error_values,
            'mse' : mse_values
        }

        # Dataframe
        inst_df = pd.DataFrame(data=per_img_data)

        # Exportar csv
        inst_df.to_csv(base_path/f'{g}_{mat}_{frame_shape}_{cmp_rate}.csv')

        exec_graphs.append(g)
        exec_mats.append(mat)
        exec_shapes.append(frame_shape)
        exec_cmp_rates.append(cmp_rate)
        exec_mean_errors.append(round(np.mean(error_values), 4))
        exec_mse_errors.append(round(np.mean(mse_values), 4))
    
    report_data = {
        'Graph': exec_graphs,
        'Matrix': exec_mats,
        'Frame Shape': exec_shapes,
        'Compression Rate': exec_cmp_rates,
        'Mean Error': exec_mean_errors,
        'Mean MSE' : exec_mse_errors
    }

    report_df = pd.DataFrame(data=report_data)
    report_df.to_csv(base_path/'MAIN_REPORT.csv')

def generate_haar_and_fourier_reports() -> None:
    '''Run the pipeline for generating full reports for the input images (Haar and Fourier basis).'''
    
    all_files = [f for f in input_dir.rglob("*.png")]
    
    frame_shape = (8, 8)
    cmp_rates = [0.02, 0.05, 0.1, 0.2]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_path = parent_dir/'reports'/f'HF ({timestamp})'

    base_path.mkdir(parents=True, exist_ok=True)
    (base_path/'images').mkdir(parents=True, exist_ok=True)

    exec_methods = []
    exec_shapes = []
    exec_cmp_rates = []
    exec_mean_errors = []
    exec_mse_errors = []


    # FOURIER
    for cmp_rate in cmp_rates:

        print('\n', '-'*10, f'Fourier {frame_shape}_{cmp_rate}', '-'*10)

        n = len(all_files)

        # Inicializa colunas do csv que será gerado
        img_names = []
        error_values = []
        mse_values = []

        for i, f in enumerate(all_files):

            name = f.name[:-4]

            print(f'\n\n[{i+1}/{n}] Arquivo {f.name}')

            frames, img, pad_shape = parse_and_segment_image(f.name, frame_shape)
            final_img = fourier_compression(frames, int(cmp_rate*frame_shape[0]*frame_shape[1]), frame_shape, pad_shape)
            final_img = final_img[:img.shape[0], :img.shape[1]]

            if name == 'fish':
                save_png_file(final_img, exp_path=(base_path/'images'/f'FOURIER_{cmp_rate}_fish'))

            error = np.linalg.norm(img - final_img) / np.linalg.norm(img)
            mse = np.mean((img - final_img) ** 2)

            img_names.append(name)
            error_values.append(round(error, 4))
            mse_values.append(round(mse, 4))
        
        per_img_data = {
            'img_name' : img_names,
            'error' : error_values,
            'mse' : mse_values
        }

        # Dataframe
        inst_df = pd.DataFrame(data=per_img_data)

        # Exportar csv
        inst_df.to_csv(base_path/f'FOURIER_{frame_shape}_{cmp_rate}.csv')

        exec_methods.append('Fourier')
        exec_shapes.append(frame_shape)
        exec_cmp_rates.append(cmp_rate)
        exec_mean_errors.append(round(np.mean(error_values), 4))
        exec_mse_errors.append(round(np.mean(mse_values), 4))

    # HAAR
    for cmp_rate in cmp_rates:

        print('\n', '-'*10, f'Haar {frame_shape}_{cmp_rate}', '-'*10)

        n = len(all_files)

        # Inicializa colunas do csv que será gerado
        img_names = []
        error_values = []
        mse_values = []

        for i, f in enumerate(all_files):

            name = f.name[:-4]

            print(f'\n\n[{i+1}/{n}] Arquivo {f.name}')

            frames, img, pad_shape = parse_and_segment_image(f.name, frame_shape)
            final_img = haar_compression(frames, int(cmp_rate*frame_shape[0]*frame_shape[1]), frame_shape, pad_shape)
            final_img = final_img[:img.shape[0], :img.shape[1]]

            if name == 'fish':
                save_png_file(final_img, exp_path=(base_path/'images'/f'FOURIER_{cmp_rate}_fish'))

            error = np.linalg.norm(img - final_img) / np.linalg.norm(img)
            mse = np.mean((img - final_img) ** 2)

            img_names.append(name)
            error_values.append(round(error, 4))
            mse_values.append(round(mse, 4))
        
        per_img_data = {
            'img_name' : img_names,
            'error' : error_values,
            'mse' : mse_values
        }

        # Dataframe
        inst_df = pd.DataFrame(data=per_img_data)

        # Exportar csv
        inst_df.to_csv(base_path/f'HAAR_{frame_shape}_{cmp_rate}.csv')

        exec_methods.append('Haar')
        exec_shapes.append(frame_shape)
        exec_cmp_rates.append(cmp_rate)
        exec_mean_errors.append(round(np.mean(error_values), 4))
        exec_mse_errors.append(round(np.mean(mse_values), 4))

    
    
    report_data = {
        'Method': exec_methods,
        'Frame Shape': exec_shapes,
        'Compression Rate': exec_cmp_rates,
        'Mean Error': exec_mean_errors,
        'Mean MSE' : exec_mse_errors
    }

    report_df = pd.DataFrame(data=report_data)
    report_df.to_csv(base_path/'MAIN_REPORT.csv')

def show_compressed_file() -> None:
    '''Run the pipeline for reading, decompressing and plotting an image from a compressed file.'''

    file_path = get_show_file_config()

    cmp_img, or_shape, pad_shape, frame_shape, graph, mat = read_img_from_file(file_path)

    M = get_associated_graph(frame_shape, graph, mat)
    e_val, e_basis = np.linalg.eigh(M)

    print('\n- REALIZANDO DESCOMPRESSÃO E RECONSTITUINDO IMAGEM')
    final_img = decompress(cmp_img, e_basis, pad_shape, or_shape, frame_shape)

    plt.figure(num=file_path.name)
    plt.imshow(final_img, cmap='gray')
    plt.axis('off')

    plt.show()







def main():

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    choice = 1

    while choice:
        choice = get_main_menu_option()

        match choice:
            case '1':
                single_file_compression()
            case '2':
                compress_all()
            case '3':
                show_compressed_file()
            case '4':
                generate_full_report()
                choice = 0
            case '5':
                generate_haar_and_fourier_reports() 
                choice = 0
            case _:
                choice = 0
    
    print('\n', '*'*10, '\tFIM DO PROGRAMA\t', '*'*10)



if __name__ == '__main__':
    main()
