import os
import sys
from pathlib import Path


def file_or_folder_picker(search_dir="output", e_type='folder'):

    base_path = Path(search_dir).resolve()

    if not base_path.exists() or not base_path.is_dir():
        print(f"Erro: '{base_path}' não é válido!")
        sys.exit(1)

    print(f'\n\tListando diretório {search_dir}:')

    if e_type == 'folder':
        all_elems = [f for f in base_path.rglob("*") if f.is_dir()]
    else:
        all_elems = [f for f in base_path.rglob('*.gft') if f.is_file()]

    if len(all_elems) == 0:
        print(f"Erro: 'DIRETÓRIO VAZIO!!")
        sys.exit(1)

    for i, elem in enumerate(all_elems):
        print(f'\t{i+1} - {elem.name}')

    choice = input('\n\tSua Escolha: ')
    while not (choice.isnumeric() and 1 <= int(choice) <= len(all_elems)):
        input('\tEscolha Inválida! Repita: ') 
    
    return all_elems[int(choice)-1]

def get_single_file_config():
    print('\n\n\t', '-'*10, 'Compressão de Arquivo Único', '-'*10)
    return {
        'img_file': input('\n\tNome do arquivo (ex: fish.png): '),
        'graph' : input('\tGrafo de representação (HAMM ou GRID): '),
        'graph_mat' : input('\tMatriz de Representação (ADJ ou LAP): '),
        'frame_shape' : input('\tDimensões das subimagens (e: 8 8): '),
        'cmp_method' : input('\tMétodo de Compressão: (THR ou KGT): '),
        'cmp_par' : input('\tParâmetro da Compressão (ex: 1e-1 ou 4): '),
        'exp_path': input('\nDigite o nome desejado para o arquivo de saída: ')
    }

def get_compress_all_config():
    print('\n\n\t', '-'*10, 'Compressão dos Arquivos de input', '-'*10)
    return {
        'graph' : input('\tGrafo de representação (HAMM ou GRID): '),
        'graph_mat' : input('\tMatriz de Representação (ADJ ou LAP): '),
        'frame_shape' : input('\tDimensões das subimagens (e: 8 8): '),
        'cmp_method' : input('\tMétodo de Compressão: (THR ou KGT): '),
        'cmp_par' : input('\tParâmetro da Compressão (ex: 1e-1 ou 4): '),
    }

def get_show_file_config():
    print('\n\n\t', '-'*10, 'Exibir Arquivo Comprimido', '-'*10)

    folder = file_or_folder_picker()

    return file_or_folder_picker(f'output/{folder.name}', 'file')

def get_report_config():
    print('\n\n\t', '-'*10, 'Relatórios de Compressão', '-'*10)

    print('\n\tTipos de relatório: ')
    print('\t1 - Comparação dos Shapes de Subdivisão')
    print('\t2 - Avaliação das Taxas de Compressão')

    return input('\nSua Escolha: ')

def get_main_menu_option():
    print('\n\n', '*'*10, 'TRANSFORMADA DE FOURIER EM GRAFOS', '*'*10)
    print('\n O que você deseja realizar?')
    print('\n1 - Comprimir um arquivo.')
    print('2 - Comprimir todos os arquivos de input.')
    print('3 - Exibir imagem comprimida.')
    print('4 - Gerar relatórios completos para os métodos propostos de compressão.')
    print('5 - Gerar relatórios completos para as bases de Fourier e Haar')
    print('\n[Digite qualquer outro comando para encerrar o programa]')  

    return input('\nSua Escolha: ')
