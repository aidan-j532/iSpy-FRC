import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# File extensions by format
MODEL_FORMATS = {
    "pytorch": [".pt"],
    "openvino": [".xml", ".bin"],
    "rknn": [".rknn"],
    "onnx": [".onnx"],
    "tflite": [".tflite"],
    "coreml": [".mlpackage"],
}

# Flatten extensions for lookup
ALL_EXTENSIONS = {}
for fmt, exts in MODEL_FORMATS.items():
    for ext in exts:
        ALL_EXTENSIONS[ext.lower()] = fmt

class ModelValidationResult:
    def __init__(self):
        self.valid_organized_models: Dict[str, Dict] = {}  # Path -> details
        self.orphan_models: Dict[str, str] = {}  # Path -> reason
        self.config_mismatches: List[Tuple[str, str, str]] = []  # (config_path, actual_path, warning)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def is_valid(self) -> bool:
        return len(self.errors) == 0 and len(self.orphan_models) == 0

    def summary(self) -> str:
        parts = []
        if self.valid_organized_models:
            parts.append(f"Valid: {len(self.valid_organized_models)}")
        if self.orphan_models:
            orphan_files = list(self.orphan_models.keys())[:2]  # Show first 2
            orphan_desc = f"{len(self.orphan_models)} ({', '.join(orphan_files)}{'...' if len(self.orphan_models) > 2 else ''})"
            parts.append(f"Orphans: {orphan_desc}")
        if self.config_mismatches:
            parts.append(f"Mismatches: {len(self.config_mismatches)}")
        if self.errors:
            error_desc = "; ".join(self.errors[:2])  # Show first 2 errors
            if len(self.errors) > 2:
                error_desc += "..."
            parts.append(f"Errors: {len(self.errors)} ({error_desc})")
        if self.warnings:
            warning_desc = "; ".join(self.warnings[:2])  # Show first 2 warnings
            if len(self.warnings) > 2:
                warning_desc += "..."
            parts.append(f"Warnings: {len(self.warnings)} ({warning_desc})")
        return ", ".join(parts) if parts else "No models found."

def validate_model_organization(repo_root: Path) -> ModelValidationResult:
    result = ModelValidationResult()
    yolo_dir = repo_root / "YoloModels"
    
    if not yolo_dir.exists():
        result.warnings.append(
            "YoloModels directory not found. Create it and add YOLO models."
        )
        return result
    
    # Find all model files
    all_model_files = {}  # extension -> [paths]
    for ext in ALL_EXTENSIONS.keys():
        all_model_files[ext] = list(yolo_dir.rglob(f"*{ext}"))
    
    # Check each model file
    for ext, model_paths in all_model_files.items():
        for model_path in model_paths:
            result_check = _validate_single_model(model_path, yolo_dir, repo_root)
            
            if result_check["valid"]:
                rel_path = str(model_path.relative_to(repo_root))
                result.valid_organized_models[rel_path] = {
                    "format": result_check["format"],
                    "size_mb": result_check["size_mb"],
                    "path": model_path,
                }
                logger.info(f"[OK] Valid model: {rel_path}")
            else:
                rel_path = str(model_path.relative_to(repo_root))
                result.orphan_models[rel_path] = result_check["reason"]
                logger.warning(f"[WARNING] Orphan model: {rel_path}")
                logger.warning(f"         {result_check['reason']}")
    
    # Check for standalone models in root or outside YoloModels
    _check_for_standalone_models(repo_root, result)
    
    return result

def _validate_single_model(model_path: Path, yolo_dir: Path, repo_root: Path) -> Dict:
    try:
        # Get extension
        ext = model_path.suffix.lower()
        fmt = ALL_EXTENSIONS.get(ext)
        
        if not fmt:
            return {
                "valid": False,
                "reason": f"Unknown model format: {ext}",
                "format": None,
                "size_mb": 0,
            }
        
        # Get file size
        size_mb = model_path.stat().st_size / (1024 * 1024)
        
        # Check if file is at least 0.01MB
        if size_mb < 0.01:
            return {
                "valid": False,
                "reason": f"Model file too small ({size_mb:.2f}MB). Possibly corrupted.",
                "format": fmt,
                "size_mb": size_mb,
            }
        
        # Check structure: YoloModels/[format]/[size]/...
        parts = model_path.relative_to(yolo_dir).parts
        
        if len(parts) < 3:
            return {
                "valid": False,
                "reason": "Model not in YoloModels/[format]/[size]/ structure",
                "format": fmt,
                "size_mb": size_mb,
            }
        
        structure_format = parts[0]
        structure_size = parts[1]
        
        # Validate format level
        valid_formats = set(MODEL_FORMATS.keys())
        if structure_format not in valid_formats:
            return {
                "valid": False,
                "reason": f"Invalid format directory: {structure_format}. "
                         f"Must be one of: {', '.join(valid_formats)}",
                "format": fmt,
                "size_mb": size_mb,
            }
        
        # Validate size level (nano, small, medium, large, etc.)
        valid_sizes = {"nano", "small", "medium", "large", "xlarge", "2xlarge"}
        if structure_size not in valid_sizes:
            return {
                "valid": False,
                "reason": f"Invalid size directory: {structure_size}. "
                         f"Must be one of: {', '.join(valid_sizes)}",
                "format": fmt,
                "size_mb": size_mb,
            }
        
        # For OpenVINO, check for .xml and .bin pair
        if fmt == "openvino":
            xml_path = model_path.parent / model_path.stem / ".xml"
            bin_path = model_path.parent / model_path.stem / ".bin"
            
            if ext == ".xml":
                expected_bin = model_path.with_suffix(".bin")
                if not expected_bin.exists():
                    return {
                        "valid": False,
                        "reason": "OpenVINO model missing .bin file (found .xml only)",
                        "format": fmt,
                        "size_mb": size_mb,
                    }
            elif ext == ".bin":
                expected_xml = model_path.with_suffix(".xml")
                if not expected_xml.exists():
                    return {
                        "valid": False,
                        "reason": "OpenVINO model missing .xml file (found .bin only)",
                        "format": fmt,
                        "size_mb": size_mb,
                    }
        
        return {
            "valid": True,
            "reason": "OK",
            "format": fmt,
            "size_mb": size_mb,
        }
    
    except Exception as e:
        return {
            "valid": False,
            "reason": f"Validation error: {str(e)}",
            "format": None,
            "size_mb": 0,
        }

def _check_for_standalone_models(repo_root: Path, result: ModelValidationResult) -> None:    
    # Check root directory
    root_models = []
    for ext in ALL_EXTENSIONS.keys():
        root_models.extend(repo_root.glob(f"*{ext}"))
    
    for model_path in root_models:
        if model_path.is_file():
            rel_path = str(model_path.relative_to(repo_root))
            result.orphan_models[rel_path] = (
                "cannot infer yolo parameters - model is standalone in repo root. "
                f"Move to YoloModels/[format]/[size]/ (e.g., YoloModels/pytorch/nano/)"
            )
            logger.warning(f"[STANDALONE] {rel_path}")
            logger.warning(f"            Cannot infer YOLO parameters for standalone model")
    
    # Check if there are model files in YoloModels root
    yolo_dir = repo_root / "YoloModels"
    if yolo_dir.exists():
        yolo_root_models = []
        for ext in ALL_EXTENSIONS.keys():
            yolo_root_models.extend(yolo_dir.glob(f"*{ext}"))
        
        for model_path in yolo_root_models:
            if model_path.is_file():
                rel_path = str(model_path.relative_to(repo_root))
                result.orphan_models[rel_path] = (
                    "cannot infer yolo parameters - model is in YoloModels root. "
                    f"Move to YoloModels/[format]/[size]/ (e.g., YoloModels/pytorch/nano/)"
                )
                logger.warning(f"[STANDALONE] {rel_path}")
                logger.warning(f"            Cannot infer YOLO parameters - move to organized directory")

def validate_config_model_paths(config: Dict, repo_root: Path, 
                               validation_result: ModelValidationResult) -> str:
    config_model_path = config.get("vision_model", {}).get("file_path", "")
    
    if not config_model_path:
        logger.warning("No model path in config.json, cannot validate vision model")
        return None
    
    config_path = Path(config_model_path)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    
    # Check if config path exists
    if config_path.exists():
        return str(config_path.relative_to(repo_root))
    
    # Try to find the model in organized structure
    model_filename = config_path.name
    yolo_dir = repo_root / "YoloModels"
    
    matches = list(yolo_dir.rglob(model_filename))
    
    if matches:
        # Use the first match (file system is source of truth)
        actual_path = matches[0]
        rel_path = str(actual_path.relative_to(repo_root))
        
        validation_result.config_mismatches.append((
            config_model_path,
            rel_path,
            f"Config path differs from filesystem. Using organized path."
        ))
        
        logger.warning(f"Config model path mismatch, specified: {config_model_path} but found: {rel_path}. Using found path.")
        
        return rel_path
    else:
        validation_result.errors.append(
            f"Model file '{model_filename}' not found. Specified: {config_model_path}"
        )
        logger.error(f"Model not found: {config_model_path}")
        return None

def _is_in_organized_structure(model_path: Path, yolo_dir: Path) -> bool:
    try:
        rel = model_path.relative_to(yolo_dir)
        parts = rel.parts
        
        if len(parts) < 3:
            return False
        
        fmt = parts[0]
        size = parts[1]
        
        valid_formats = set(MODEL_FORMATS.keys())
        valid_sizes = {"nano", "small", "medium", "large", "xlarge", "2xlarge"}
        
        return fmt in valid_formats and size in valid_sizes
    
    except ValueError:
        return False

def enforce_model_organization(repo_root: Path, config: Dict) -> Tuple[bool, str]:
    # Validate filesystem organization
    validation_result = validate_model_organization(repo_root)
    
    # Log results
    summary = validation_result.summary()
    if summary:
        logger.info(summary)
    
    # Validate config paths
    corrected_path = validate_config_model_paths(config, repo_root, validation_result)
    
    # Report final status
    if validation_result.errors:
        logger.error("Model validation failed with errors")
        return False, None
    
    if corrected_path:
        # logger.info(f"Using model: {corrected_path}")
        return True, corrected_path
    else:
        logger.error("No valid model path determined")
        return False, None

def cmd_validate_models() -> int:
    result = validate_model_organization(Path.cwd())
    
    print("\n" + "="*70)
    print("YOLO Model Validation Report".center(70))
    print("="*70)
    print(result.summary())
    print("="*70 + "\n")
    
    if result.is_valid():
        logger.info("All models are properly organized.")
        return 0
    else:
        if result.orphan_models:
            logger.warning("Found orphan/standalone models - cannot infer YOLO parameters")
            logger.info("Move models to: YoloModels/[format]/[size]/")
        if result.errors:
            logger.error("Validation failed with errors")
            return 1
        return 0

def cmd_check_organization() -> int:
    result = validate_model_organization(Path.cwd())
    
    print("\nYoloModels Organization Status:")
    print("-" * 50)
    
    if result.valid_organized_models:
        print(f"Valid organized models: {len(result.valid_organized_models)}")
        for path in result.valid_organized_models:
            print(f"  + {path}")
    else:
        print("No valid organized models found")
    
    if result.orphan_models:
        print(f"\nOrphan models (not in organized structure): {len(result.orphan_models)}")
        for path, reason in result.orphan_models.items():
            print(f"  - {path}")
            print(f"    {reason}")
    
    print("\nCorrect Structure:")
    print("  YoloModels/[format]/[size]/model_file")
    print("  Examples:")
    print("    YoloModels/pytorch/nano/yolov8n.pt")
    print("    YoloModels/openvino/nano/yolov8n.xml")
    print("    YoloModels/rknn/nano/yolov8n.rknn")
    print()
    
    return 0

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ["validate-models", "check-org"]:
        if sys.argv[1] == "validate-models":
            sys.exit(cmd_validate_models())
        else:
            sys.exit(cmd_check_organization())
    else:
        print("Model Validation CLI")
        print("Usage: python -m VisionCore.validations.model_validator validate-models")
        print("       python -m VisionCore.validations.model_validator check-org")