# Reciclapp Image Verification Demo

Projeto demonstrativo para validar imagens de materiais no fluxo do Reciclapp. Nesta versão não há API, Docker nem testes automatizados obrigatórios: você coloca imagens em uma pasta, executa comandos no terminal e o algoritmo gera relatórios JSON em disco.

## Objetivo

O projeto simula o serviço que analisaria uma imagem enviada por um usuário antes de ela ser exibida para um coletor.

Fluxo executado:

1. Validação técnica da imagem.
2. Análise de qualidade com OpenCV.
3. Moderação com Google Cloud Vision SafeSearch ou simulação local.
4. Classificação com TensorFlow/Keras ou classificador demonstrativo.
5. Decisão final: `APROVADA`, `REPROVADA`, `REVISAO_MANUAL` ou `SOLICITAR_NOVA_IMAGEM`.

## Categorias

O classificador trabalha com estas categorias:

```text
PLASTICO
PAPEL
VIDRO
METAL
ELETRONICO
ORGANICO
GRANDE_PORTE
LIXO_GERAL
FORA_DE_CONTEXTO
```

## Ferramentas

| Ferramenta | Uso no projeto |
| --- | --- |
| Python 3.11+ / 3.13 | Execução dos scripts locais. |
| Pillow | Abertura, validação e conversão de imagens para RGB. |
| OpenCV | Cálculo de brilho médio e desfoque por Laplaciano. |
| NumPy | Manipulação de arrays de imagem. |
| Pydantic | Padronização dos objetos de resposta. |
| python-dotenv | Leitura das configurações do arquivo `.env`. |
| Google Cloud Vision | SafeSearch real, quando habilitado. |
| TensorFlow/Keras | Treinamento e inferência do classificador real. |
| KaggleHub | Download de datasets públicos prontos. |
| Matplotlib | Gráfico do histórico de treinamento. |
| scikit-learn | Matriz de confusão na avaliação. |

## Estrutura

```text
app/
├── main.py
├── config.py
├── schemas/
├── services/
├── models/
└── utils/
input_images/
output_reports/
training/
├── dataset/
├── prepare_ready_dataset.py
├── train_model.py
└── evaluate_model.py
requirements.txt
.env.example
```

## Instalação

Na raiz do projeto:

```bash
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

No Windows PowerShell, a ativação costuma ser:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Configuração

Arquivo `.env` recomendado para apresentação offline:

```env
USE_GOOGLE_SAFESEARCH=false
CLASSIFIER_DEMO_MODE=true
INPUT_IMAGES_DIR=input_images
OUTPUT_REPORTS_DIR=output_reports
IMAGE_MODEL_PATH=app/models/recyclable_model.keras
```

Com essa configuração:

- O SafeSearch é simulado como seguro.
- Se o modelo Keras não existir, o classificador demonstrativo é usado.
- O fluxo completo roda sem credenciais Google.

Para usar modelo real treinado:

```env
CLASSIFIER_DEMO_MODE=false
```

Para usar SafeSearch real:

```env
USE_GOOGLE_SAFESEARCH=true
GOOGLE_APPLICATION_CREDENTIALS=/caminho/absoluto/google-vision-service-account.json
```

## Comandos De Sanidade

Verificar sintaxe dos módulos:

```bash
python -m compileall app training
```

Verificar se o CLI responde:

```bash
python -m app.main --help
```

Verificar quantidade de imagens por classe:

```bash
for d in training/dataset/*; do [ -d "$d" ] || continue; printf '%s ' "${d##*/}"; find "$d" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' \) | wc -l; done
```

## Preparar Dataset Pronto

Dataset principal usado:

```text
glhdamar/new-trash-classfication-dataset
```

Dataset complementar para `GRANDE_PORTE`:

```text
udaysankarmukherjee/furniture-image-dataset
```

Baixar e preparar tudo:

```bash
python training/prepare_ready_dataset.py --max-per-class 300 --include-grande-porte
```

Baixar somente imagens de `GRANDE_PORTE`:

```bash
python training/prepare_ready_dataset.py --only-grande-porte --max-per-class 100
```

Usar dataset local já baixado:

```bash
python training/prepare_ready_dataset.py --source-dir /caminho/do/dataset --max-per-class 300
```

Mapeamento aplicado:

```text
plastic           -> plastico
paper/cardboard   -> papel
glass             -> vidro
metal             -> metal
e-waste           -> eletronico
organic           -> organico
large/furniture   -> grande_porte
trash/garbage     -> lixo_geral
textile/clothes   -> fora_de_contexto
```

## Treinamento

Treinar o modelo:

```bash
python training/train_model.py
```

Arquivos gerados:

```text
app/models/recyclable_model.keras
app/models/recyclable_labels.json
training/training_history.png
```

Importante: se novas categorias forem adicionadas ou imagens forem alteradas, treine novamente. O modelo antigo não aprende categorias novas sozinho.

## Avaliação

Avaliar o modelo treinado:

```bash
python training/evaluate_model.py
```

Saída esperada:

```text
loss=...
accuracy=...
labels=[...]
confusion_matrix=
...
```

## Execução Do Pipeline

Processar uma imagem específica:

```bash
python -m app.main --image input_images/plastico.jpg --material PLASTICO --collection-point-id ponto-001
```

Processar todas as imagens em `input_images/`:

```bash
python -m app.main --material PLASTICO --collection-point-id ponto-001
```

Os relatórios são salvos em:

```text
output_reports/
```

## Testes Manuais

Antes dos testes, coloque imagens em `input_images/`. No modo demonstrativo, o nome do arquivo influencia a classe detectada.

### Teste 1: Aprovação

```bash
python -m app.main --image input_images/plastico.jpg --material PLASTICO
```

Resultado esperado:

```text
Status: APROVADA
Etapa: FINAL_DECISION
```

### Teste 2: Material Diferente Do Informado

```bash
python -m app.main --image input_images/plastico.jpg --material METAL
```

Resultado esperado:

```text
Status: REVISAO_MANUAL
Motivo: Material detectado diferente do material informado
```

### Teste 3: Fora De Contexto

```bash
python -m app.main --image input_images/fora_de_contexto.jpg --material PLASTICO
```

Resultado esperado:

```text
Status: REPROVADA
Etapa: RECYCLABLE_CLASSIFICATION
```

### Teste 4: Grande Porte

```bash
python -m app.main --image input_images/sofa_grande_porte.jpg --material GRANDE_PORTE
```

Resultado esperado:

```text
Status: APROVADA
Material detectado: GRANDE_PORTE
```

### Teste 5: Lixo Geral

```bash
python -m app.main --image input_images/lixo_geral.jpg --material LIXO_GERAL
```

Resultado esperado:

```text
Status: APROVADA
Material detectado: LIXO_GERAL
```

### Teste 6: Qualidade Ruim

Use uma imagem muito escura ou muito borrada:

```bash
python -m app.main --image input_images/imagem_escura.jpg --material PLASTICO
```

Resultado esperado:

```text
Status: SOLICITAR_NOVA_IMAGEM
Etapa: IMAGE_QUALITY
```

### Teste 7: Formato Inválido

Tente usar um arquivo que não seja imagem:

```bash
python -m app.main --image input_images/arquivo.txt --material PLASTICO
```

Resultado esperado:

```text
Status: REPROVADA
Etapa: TECHNICAL_VALIDATION
```

### Teste 8: Processar Várias Imagens

Coloque várias imagens em `input_images/` e rode:

```bash
python -m app.main --material PLASTICO
```

Cada imagem gera um relatório próprio em `output_reports/`.

## Classificador Demonstrativo

Quando `CLASSIFIER_DEMO_MODE=true`, o sistema tenta classificar pela pista no nome do arquivo:

```text
plastico.jpg, garrafa_pet.jpg       -> PLASTICO
papel.jpg, papelao.jpg              -> PAPEL
vidro.jpg                           -> VIDRO
lata_metal.jpg                      -> METAL
celular_eletronico.jpg              -> ELETRONICO
organico.jpg                        -> ORGANICO
sofa_grande_porte.jpg, entulho.jpg  -> GRANDE_PORTE
lixo_geral.jpg, rejeito.jpg         -> LIXO_GERAL
fora_de_contexto.jpg                -> FORA_DE_CONTEXTO
```

Se não houver pista no nome, ele usa uma heurística simples por cor média. Essa heurística existe apenas para apresentação.

## Exemplo De Relatório

```json
{
  "collection_point_id": "ponto-001",
  "status": "APROVADA",
  "stage": "FINAL_DECISION",
  "reason": "Imagem aprovada",
  "material_informado": "PLASTICO",
  "material_detectado": "PLASTICO",
  "confidence": 0.88,
  "quality": {
    "brightness_score": 120.5,
    "blur_score": 230.8,
    "quality_status": "BOA",
    "reason": "Imagem com qualidade suficiente"
  },
  "safe_search": {
    "adult": "VERY_UNLIKELY",
    "spoof": "UNLIKELY",
    "medical": "VERY_UNLIKELY",
    "violence": "VERY_UNLIKELY",
    "racy": "VERY_UNLIKELY"
  },
  "classification": {
    "predicted_class": "PLASTICO",
    "confidence": 0.88,
    "scores_by_class": {
      "PLASTICO": 0.88,
      "PAPEL": 0.015,
      "VIDRO": 0.015,
      "METAL": 0.015,
      "ELETRONICO": 0.015,
      "ORGANICO": 0.015,
      "GRANDE_PORTE": 0.015,
      "LIXO_GERAL": 0.015,
      "FORA_DE_CONTEXTO": 0.015
    }
  }
}
```

## Roteiro De Apresentação

1. Mostrar as pastas `input_images/`, `output_reports/` e `training/dataset/`.
2. Explicar as ferramentas usadas: Pillow, OpenCV, Google Vision, TensorFlow/Keras e Pydantic.
3. Rodar um teste aprovado.
4. Rodar um teste com material divergente.
5. Rodar um teste com `GRANDE_PORTE` ou `LIXO_GERAL`.
6. Abrir o JSON gerado em `output_reports/`.
7. Explicar que o mesmo fluxo pode usar SafeSearch real e modelo treinado real.
