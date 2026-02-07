from setuptools import setup, find_packages
from pathlib import Path

def parse_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
        return [line for line in lines if line and not line.startswith('#')]

here = Path(__file__).parent.resolve()

req_path = here / "requirements.txt"

setup(
    name="crop_for_trust",
    version="0.1.0",
    description="Crop for Trust: Reliability-Aware Visual Debiasing Pipeline",
    author="KyeongHa Hwang",
    
    packages=find_packages(), 
    
    python_requires='>=3.10',
    
    install_requires=parse_requirements(req_path) if req_path.exists() else [],
)