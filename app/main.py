import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from app.config import get_settings
from app.schemas.request_schema import MaterialType
from app.services.pipeline_service import ImageAnalysisPipeline


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format='{"level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Executa a verificacao local de imagens do Reciclapp."
    )
    parser.add_argument(
        "--image",
        help="Caminho de uma imagem especifica. Se omitido, processa input_images/.",
    )
    parser.add_argument(
        "--material",
        required=True,
        choices=[material.value for material in MaterialType],
        help="Material informado pelo usuario.",
    )
    parser.add_argument(
        "--collection-point-id",
        default="apresentacao-001",
        help="Identificador do ponto de coleta.",
    )
    return parser.parse_args()


def list_images(input_dir: Path) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    return sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_report(output_dir: Path, image_path: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{image_path.stem}_resultado.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    load_dotenv()
    configure_logging()
    settings = get_settings()
    args = parse_args()

    image_paths = [Path(args.image)] if args.image else list_images(Path(settings.input_images_dir))
    if not image_paths:
        print(f"Nenhuma imagem encontrada em {settings.input_images_dir}/")
        return 1

    pipeline = ImageAnalysisPipeline(settings)
    output_dir = Path(settings.output_reports_dir)

    print(f"Ferramentas usadas: Pillow, OpenCV, NumPy, Pydantic, python-dotenv")
    print("Integrações disponíveis: Google Cloud Vision SafeSearch, TensorFlow/Keras")
    print(f"SafeSearch Google ativo: {settings.use_google_safesearch}")
    print(f"Classificador demonstrativo ativo se modelo faltar: {settings.classifier_demo_mode}")
    print()

    for image_path in image_paths:
        result = pipeline.analyze_file(
            image_path=image_path,
            material_type=args.material,
            collection_point_id=args.collection_point_id,
        )
        payload = result.model_dump(mode="json")
        report_path = write_report(output_dir, image_path, payload)

        print(f"Imagem: {image_path}")
        print(f"Status: {result.status}")
        print(f"Etapa: {result.stage}")
        print(f"Motivo: {result.reason}")
        print(f"Relatorio: {report_path}")
        print("-" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
