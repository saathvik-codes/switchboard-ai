"""
AWS Lambda entry point (template.yaml): the FastAPI app behind API Gateway through the Mangum adapter. The modules
import each other by bare name, so their folders go on the path first, as api/main.py does.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in ("ml", "agent", "api"):
    sys.path.insert(0, str(ROOT / sub))

from mangum import Mangum  # noqa: E402
from main import app       # noqa: E402

handler = Mangum(app, lifespan="off")
