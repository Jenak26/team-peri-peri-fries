git init
mkdir -p peri/core train api web/vendor artifacts evidence tools tests data/authentic data/corpus docs/superpowers/plans
New-Item -ItemType File -Force peri/__init__.py
New-Item -ItemType File -Force peri/core/__init__.py
New-Item -ItemType File -Force train/__init__.py
New-Item -ItemType File -Force api/__init__.py
New-Item -ItemType File -Force tools/__init__.py
New-Item -ItemType File -Force tests/__init__.py
New-Item -ItemType File -Force data/authentic/.gitkeep
New-Item -ItemType File -Force data/corpus/.gitkeep

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
