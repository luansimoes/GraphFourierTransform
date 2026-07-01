# Compressão de Imagens por GFT

Esse repositório contém uma implementação da transformada de Fourier em grafos (GFT) para compressão de imagens. 

Primeiro, instale as dependências com:
```
pip install -r requirements.txt
```

Para iniciar o programa, execute:
```
python -m gft.cli
```

As principais funcionalidades são:

1. Comprimir um arquivo único da pasta de inputs com uma dada configuração de compressão, salvando arquivo .gft comprimido e exibindo a imagem final.
2. Executar um método de compressão por transformada de Fourier em grafos para todas as imagens na pasta input.
3. Descomprimir e exibir uma imagem comprimida na pasta output.
4. Gerar relatórios de compressão para as instâncias de input, com as combinações de configurações definidas no código:
    1. Conf. 1: Executa para diversos shapes de subimagens.
    2. Conf. 2: Executa para diversas taxas de compressão (2%, 5%, 10% e 20%).
5. Gera relatórios para a compressão tradicional por wavelets de Haar e pela FFT.

Os parâmetros ajustáveis de compressão são: 

* Grafo associado:
    - GRID => Grafo grade
    - HAMM => Grafo de Hamming H(2, 3)

* Matriz de Representação:
    - ADJ => Matriz de Adjacência
    - LAP => Matriz Laplaciana

* Método de Compressão:
    - THR => Zera coeficientes abaixo de um limiar (threshold) definido.
    - KGT => Preserva apenas as frequências associadas aos k maiores coeficientes do sinal na base de autovetores.
