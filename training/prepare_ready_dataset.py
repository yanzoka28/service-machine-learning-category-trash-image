import argparse
import shutil
from collections import Counter
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_DATASET = "glhdamar/new-trash-classfication-dataset"
DEFAULT_GRANDE_PORTE_DATASET = "udaysankarmukherjee/furniture-image-dataset"
OUTPUT_DIR = Path("training/dataset")

CLASS_ALIASES = {
    "PLASTICO": {"plastic", "plastico", "plastics"},
    "PAPEL": {"paper", "papel", "cardboard", "carton", "papelao"},
    "VIDRO": {"glass", "vidro", "green-glass", "brown-glass", "white-glass"},
    "METAL": {"metal", "metals", "aluminium", "aluminum"},
    "ELETRONICO": {"e-waste", "ewaste", "e_waste", "electronic", "electronics", "battery"},
    "ORGANICO": {"organic", "organico", "biological", "food organics", "vegetation"},
    "GRANDE_PORTE": {
        "bulky",
        "large",
        "large-waste",
        "furniture",
        "sofa",
        "mattress",
        "appliance",
        "construction",
        "demolition",
        "entulho",
        "grande porte",
    },
    "LIXO_GERAL": {
        "trash",
        "garbage",
        "general",
        "general-waste",
        "residual",
        "rejeito",
        "lixo geral",
        "mixed",
    },
    "FORA_DE_CONTEXTO": {
        "miscellaneous trash",
        "misc",
        "textile",
        "clothes",
        "shoes",
    },
}

FOLDER_BY_CLASS = {
    "PLASTICO": "plastico",
    "PAPEL": "papel",
    "VIDRO": "vidro",
    "METAL": "metal",
    "ELETRONICO": "eletronico",
    "ORGANICO": "organico",
    "GRANDE_PORTE": "grande_porte",
    "LIXO_GERAL": "lixo_geral",
    "FORA_DE_CONTEXTO": "fora_de_contexto",
}


def normalize_name(value: str) -> str:
    return (
        value.lower()
        .strip()
        .replace("_", "-")
        .replace(" ", "-")
        .replace("á", "a")
        .replace("à", "a")
        .replace("ã", "a")
        .replace("â", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("õ", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )


def detect_class(path: Path) -> str | None:
    parts = [normalize_name(part) for part in path.parts]
    for class_name, aliases in CLASS_ALIASES.items():
        normalized_aliases = {normalize_name(alias) for alias in aliases}
        if any(part in normalized_aliases for part in parts):
            return class_name
    return None


def image_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def prepare_output(clean: bool):
    if clean and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for folder in FOLDER_BY_CLASS.values():
        target = OUTPUT_DIR / folder
        target.mkdir(parents=True, exist_ok=True)
        (target / ".gitkeep").touch()


def copy_dataset(source_dir: Path, max_per_class: int, clean: bool) -> Counter:
    if source_dir.resolve() == OUTPUT_DIR.resolve():
        raise SystemExit(
            "A origem não pode ser o próprio training/dataset para preparar o dataset completo. "
            "Use --only-grande-porte se quiser apenas completar a categoria grande_porte."
        )

    prepare_output(clean)
    counters: Counter[str] = Counter()
    skipped = 0

    for source_file in image_files(source_dir):
        class_name = detect_class(source_file.relative_to(source_dir))
        if class_name is None:
            skipped += 1
            continue
        if counters[class_name] >= max_per_class:
            continue

        target_folder = OUTPUT_DIR / FOLDER_BY_CLASS[class_name]
        target_file = target_folder / f"{counters[class_name]:05d}_{source_file.name}"
        shutil.copy2(source_file, target_file)
        counters[class_name] += 1

    print("Resumo do dataset preparado:")
    for class_name, folder in FOLDER_BY_CLASS.items():
        print(f"- {folder}: {counters[class_name]} imagens")
    print(f"- ignoradas sem mapeamento: {skipped}")
    return counters


def copy_grande_porte_dataset(source_dir: Path, max_images: int) -> int:
    target_folder = OUTPUT_DIR / FOLDER_BY_CLASS["GRANDE_PORTE"]
    target_folder.mkdir(parents=True, exist_ok=True)
    (target_folder / ".gitkeep").touch()

    copied = 0
    for source_file in image_files(source_dir):
        if copied >= max_images:
            break
        target_file = target_folder / f"{copied:05d}_{source_file.name}"
        shutil.copy2(source_file, target_file)
        copied += 1

    print(f"- grande_porte complementar: {copied} imagens")
    return copied


def download_from_kaggle(dataset: str) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "Instale as dependencias primeiro: pip install -r requirements.txt"
        ) from exc

    downloaded_path = kagglehub.dataset_download(dataset)
    return Path(downloaded_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baixa e reorganiza um dataset pronto para o treinamento."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Slug do dataset Kaggle. Ex: glhdamar/new-trash-classfication-dataset",
    )
    parser.add_argument(
        "--source-dir",
        help="Diretorio local de um dataset ja baixado. Se informado, nao baixa do Kaggle.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=700,
        help="Quantidade maxima de imagens por classe para deixar o treino mais leve.",
    )
    parser.add_argument(
        "--include-grande-porte",
        action="store_true",
        help="Baixa dataset complementar de moveis/eletrodomesticos para grande_porte.",
    )
    parser.add_argument(
        "--only-grande-porte",
        action="store_true",
        help="Baixa/copia somente imagens para a categoria grande_porte.",
    )
    parser.add_argument(
        "--grande-porte-dataset",
        default=DEFAULT_GRANDE_PORTE_DATASET,
        help="Slug Kaggle complementar para grande_porte.",
    )
    parser.add_argument(
        "--grande-porte-source-dir",
        help="Diretorio local complementar para grande_porte.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Nao limpa training/dataset antes de copiar.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.only_grande_porte:
        source_dir = Path(args.source_dir) if args.source_dir else download_from_kaggle(args.dataset)
        print(f"Dataset origem: {source_dir}")
        copy_dataset(source_dir, max_per_class=args.max_per_class, clean=not args.no_clean)

    if args.include_grande_porte or args.only_grande_porte:
        grande_porte_source = (
            Path(args.grande_porte_source_dir)
            if args.grande_porte_source_dir
            else download_from_kaggle(args.grande_porte_dataset)
        )
        print(f"Dataset grande_porte origem: {grande_porte_source}")
        copy_grande_porte_dataset(grande_porte_source, max_images=args.max_per_class)


if __name__ == "__main__":
    main()
